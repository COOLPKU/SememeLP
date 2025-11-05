import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertModel, BertTokenizer
# from transformers import BertTokenizer, BertModel
from transformers.activations import GELUActivation
import copy
import torch
from utils import get_sim, data_to_device
import tqdm
import random
import math
# from optimum.bettertransformer import BetterTransformer
from utils import negative_sampling_Entities, Entities, Konwledge_Graph, data_to_device, batch_split, get_sim, get_logger, set_seed, collate

def _get_sememe_feature(hw_model,token_ids, mask):
    hw_model.eval()
    with torch.no_grad():
        cls_a_hidden, cls_b_hidden, multi_logits, single_logits = hw_model.predict(token_ids, mask)
    return cls_a_hidden.detach(),cls_b_hidden.detach()

class MoEFusion(nn.Module):
    def __init__(self, embed_dim=768, num_experts=12, top_k=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k  # 激活的专家数

        # 定义多个专家，每个专家是一个独立的门控网络
        self.experts = nn.ModuleList([
            self._build_expert() for _ in range(num_experts)
        ])
        
        # 门控网络，动态选择专家
        self.gate = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def _build_expert(self):
        # 每个专家是一个独立的门控网络（可定制更复杂结构）
        return nn.Sequential(
            nn.Linear(2*self.embed_dim, 4*self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(4*self.embed_dim, self.embed_dim),
            nn.Sigmoid()
        )

    def forward(self, combined):
        
        
        # 计算门控权重 [batch, num_experts]
        gate_weights = self.gate(combined)
        
        # 选择 top-k 专家（稀疏激活）
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # 归一化
        
        # 计算各专家输出并加权融合
        expert_outputs = []
        for expert in self.experts:
            expert_gate = expert(combined)  # [batch, embed_dim]
            expert_outputs.append(expert_gate)
        
        # 按 top-k 索引收集输出并加权 [batch, top_k, embed_dim]
        selected_outputs = torch.stack(expert_outputs, dim=1).gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )
        # 加权求和 [batch, embed_dim]
        fused_gate = (selected_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)
        
       
        return fused_gate

class DualSememeFusion(nn.Module):
    def __init__(self, embed_dim=768,mode='hr'):
        super().__init__()
        # 阶段1：独立门控层
        
        self.embed_dim = embed_dim
        
        #self.gate_main = self._build_gate(self.embed_dim)
        #self.gate_all = self._build_gate(self.embed_dim)

        
        
        
        # 阶段3：联合融合层
        self.joint_fuse = nn.Sequential(
            nn.Linear(3*self.embed_dim, self.embed_dim*6),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim*6, self.embed_dim),
        )
        
        # 初始化
        self._init_weights()

    

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text_embed, main_sememe, all_sememe):
        """
        输入维度: [batch, embed_dim]
        """
        # === 阶段1：独立门控 ===
        # Main 义原融合
        combined_main = torch.cat([text_embed, main_sememe,all_sememe], dim=-1)
        fuse = self.joint_fuse(combined_main)
        #fused_main = gate_main * text_embed + (1 - gate_main) * main_sememe  # [batch, dim]
        
        

        # === 阶段3：联合优化 ===
        #joint_input = torch.cat([text_embed, fused_main], dim=-1)
        #final_gate = self.joint_fuse(joint_input)
        
        return fuse






            
    







class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, pre_seq_len, hidden_size, prefix_hidden_size, num_hidden_layers, prompt_len=8, prefix_projection=True):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prefix_hidden_size, prompt_len * num_hidden_layers * 2 * hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(pre_seq_len, prompt_len * num_hidden_layers * 2 * hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

   
class neighborhood_embedding(torch.nn.Module):

    def __init__(self, args, entities_size, relations_size, NeighborsRepresentationSets, plm_args):
        super().__init__()
        self.args = args
        self.NeighborsRepresentationSets = NeighborsRepresentationSets
        self.entities_size = entities_size
        self.relations_size = relations_size
        self.hidden_size = plm_args.hidden_size
        self.prefix_hidden_size = plm_args.hidden_size
        self.num_hidden_layers = plm_args.num_hidden_layers
        self.num_attention_heads = plm_args.num_attention_heads
        self.r_prompt_len = args.r_prompt_len
        self.hidden_dropout_prob = plm_args.hidden_dropout_prob
        
        self.CLS = torch.nn.Parameter(torch.empty(1, args.soft_prompt_len*self.hidden_size), requires_grad=True)
        xavier_normal_(self.CLS.data)
        # self.r0 = torch.nn.Parameter(torch.empty(1, self.hidden_size), requires_grad=True)
        # xavier_normal_(self.r0.data)
        if args.r_prompt_len>0:
            self.R_prompt = PrefixEncoder(relations_size, self.hidden_size, self.prefix_hidden_size, self.num_hidden_layers, prompt_len=self.r_prompt_len)
            self.R_prompt_dropout = torch.nn.Dropout(self.hidden_dropout_prob)
            # self.R_prompt = torch.nn.Embedding(relations_size, self.r_prompt_len*self.hidden_size)
            # xavier_normal_(self.R_prompt.weight.data)
        self.R_emb = torch.nn.Embedding(relations_size+1, self.hidden_size)
        xavier_normal_(self.R_emb.weight.data)

    def get_r_prompt(self, r_idx):
        r_prompt = self.R_prompt(r_idx).view(r_idx.size(0), -1, self.hidden_size)
        past_key_values = r_prompt.view(
            r_idx.size(0),
            self.r_prompt_len,
            self.num_hidden_layers * 2, 
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads
        )
        past_key_values = self.R_prompt_dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) # layers, bs, num_heads, seq_len, hid_dim
        prefix_mask = torch.ones(r_idx.size(0), self.r_prompt_len).long().to(r_idx.device)
        return past_key_values, prefix_mask

    def forward(
            self, 
            E_emb:torch.Tensor, 
            h_idx:torch.Tensor, 
            r_idx:torch.Tensor=None, 
            sample_neighborhood:torch.Tensor=None, 
            sample_neighborhood_mask:torch.Tensor=None, 
            mask_idx:torch.Tensor=None
        ):
        batch_size = h_idx.size(0)
        device = h_idx.device

        
        CLS = self.CLS.view(1, self.args.soft_prompt_len, self.hidden_size).expand(batch_size, self.args.soft_prompt_len, self.hidden_size)
        # if r_idx is not None:
        #     r = self.R_prompt(r_idx).view(batch_size, -1, self.hidden_size)
        #     outputs_embeds = torch.concat((CLS, r), dim=1)
        #     attention_mask = torch.ones((batch_size, CLS.size(1)+r.size(1)), device=device, dtype=torch.long)
        # else:
        outputs_embeds = CLS
        attention_mask = torch.ones((batch_size, CLS.size(1)), device=device, dtype=torch.long)
        # outputs_embeds = torch.empty((batch_size, 0, self.hidden_size), device=device, dtype=E_emb.dtype)
        # attention_mask = torch.ones((batch_size, outputs_embeds.size(1)), device=device, dtype=torch.long)
        if sample_neighborhood is not None:
            neighbors = E_emb.data[sample_neighborhood[:,:,0].to(E_emb.data.device).view(-1)].view(batch_size, -1, self.hidden_size).to(device)
            neighbor_relations = self.R_emb(sample_neighborhood[:,:,1].view(-1)).view(batch_size, -1, self.hidden_size)
            outputs_embeds = torch.concat((outputs_embeds, neighbors + neighbor_relations), dim=1)
            attention_mask = torch.concat((attention_mask, sample_neighborhood_mask), dim=1)
        token_type_ids = torch.zeros_like(attention_mask, dtype=torch.long).to(device)
        position_ids = token_type_ids[0:1,:] + 511

        return outputs_embeds, attention_mask, token_type_ids, position_ids

class MoCo_encoder(nn.Module):

    def __init__(self, args, entities_size, relations_size, NeighborsRepresentationSets, method='MLP', hr_encoder=False,mode='hr'):
        '''
        method:'concat', 'linear', 'MLP'
        output_type:'CLS', 'mean'
        '''
        super().__init__()
        self.args = args
        self.entities_size = entities_size
        self.relations_size = relations_size
        self.plm_args = AutoConfig.from_pretrained(args.plm_name)
        self.hidden_size = self.plm_args.hidden_size
        self.mode =mode

        #config = Config(self.hidden_size+args.hw_feature_size)

        self.dualSememeFusion = DualSememeFusion(self.hidden_size,self.mode)


        self.r_prompt_len = args.r_prompt_len
        self.encoder:BertModel = BertModel.from_pretrained(args.plm_name)
        self.neighborhood_Embedding = neighborhood_embedding(args, 
                                                             entities_size, 
                                                             relations_size, 
                                                             NeighborsRepresentationSets, 
                                                             self.plm_args)
        self.method = method
        if  self.method == 'linear':
            self.linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.linear.weight.data = torch.zeros_like(self.linear.weight.data)
            self.linear.bias.data = torch.mean(self.encoder.embeddings.word_embeddings.weight, dim=0)
        elif  self.method == 'MLP':
            self.MLP = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size*2),
                nn.Tanh(),
                nn.Linear(self.hidden_size*2, self.hidden_size),
            )
            # self.MLP[-1].weight.data = torch.zeros_like(self.MLP[-1].weight.data)
            # self.MLP[-1].bias.data = torch.mean(self.encoder.embeddings.word_embeddings.weight, dim=0)
        else:
            raise ValueError('')
    
    def set_E_emb(self, E_emb):
        self.E_emb = E_emb

    def concat_encoding(self, text_dict, neighborhood_dict, r_idx=None):
        inputs_embeds = torch.concat((text_dict['inputs_embeds'], neighborhood_dict['inputs_embeds']), dim=1)
        attention_mask = torch.concat((text_dict['attention_mask'], neighborhood_dict['attention_mask']), dim=1)
        token_type_ids = torch.concat((text_dict['token_type_ids'], neighborhood_dict['token_type_ids']), dim=1)
        position_ids = torch.concat((text_dict['position_ids'], neighborhood_dict['position_ids']), dim=1)

        if r_idx is not None and self.args.r_prompt_len>0:
            past_key_values, prefix_mask = self.neighborhood_Embedding.get_r_prompt(r_idx)
            _attention_mask = torch.concat((prefix_mask, attention_mask), dim=1)
        else:
            _attention_mask = attention_mask
            past_key_values = None

        last_hidden_state = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            past_key_values = past_key_values
        ).last_hidden_state

        output_vector = last_hidden_state_pooling(last_hidden_state, attention_mask)

        return output_vector
    
    def linear_encoding(self, text_dict, neighborhood_dict, r_idx=None):
        inputs_embeds = self.linear(neighborhood_dict['inputs_embeds'])
        inputs_embeds = torch.concat((text_dict['inputs_embeds'], inputs_embeds), dim=1)
        attention_mask = torch.concat((text_dict['attention_mask'], neighborhood_dict['attention_mask']), dim=1)
        token_type_ids = torch.concat((text_dict['token_type_ids'], neighborhood_dict['token_type_ids']), dim=1)
        position_ids = torch.concat((text_dict['position_ids'], neighborhood_dict['position_ids']), dim=1)

        if r_idx is not None:
            past_key_values, prefix_mask = self.neighborhood_Embedding.get_r_prompt(r_idx)
            _attention_mask = torch.concat((prefix_mask, attention_mask), dim=1)
        else:
            _attention_mask = attention_mask
            past_key_values = None


        last_hidden_state = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            past_key_values = past_key_values
        ).last_hidden_state

        output_vector = last_hidden_state_pooling(last_hidden_state, attention_mask)

        return output_vector
    
    def MLP_encoding(self, text_dict, neighborhood_dict, r_idx):

        inputs_embeds = self.MLP(neighborhood_dict['inputs_embeds'])
        inputs_embeds = torch.concat((text_dict['inputs_embeds'], inputs_embeds), dim=1)
        attention_mask = torch.concat((text_dict['attention_mask'], neighborhood_dict['attention_mask']), dim=1)
        token_type_ids = torch.concat((text_dict['token_type_ids'], neighborhood_dict['token_type_ids']), dim=1)
        position_ids = torch.concat((text_dict['position_ids'], neighborhood_dict['position_ids']), dim=1)

        if r_idx is not None and self.args.r_prompt_len>0:
            past_key_values, prefix_mask = self.neighborhood_Embedding.get_r_prompt(r_idx)
            attention_mask = torch.concat((prefix_mask, attention_mask), dim=1)
        else:
            past_key_values = None

        last_hidden_state = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            past_key_values=past_key_values
        ).last_hidden_state


        if neighborhood_dict.get('attention_mask', None) is not None:
            # output_mask = torch.concat((text_dict['attention_mask'], torch.zeros_like(neighborhood_dict['attention_mask'])), dim=1)
            output_mask = torch.concat((text_dict['attention_mask'], neighborhood_dict['attention_mask']), dim=1)
        else:
            output_mask = text_dict['attention_mask']
        output_vector = last_hidden_state_pooling(last_hidden_state, output_mask)

        return output_vector
 
    def forward(self, hw_model,input_ids, attention_mask, token_type_ids,hw_input_ids, hw_attention_mask, hw_token_type_ids, h_idx=None, r_idx=None, sample_neighborhood=None, sample_neighborhood_mask=None, mask_idx=None):

        text_dict = {}
        text_dict['inputs_embeds'] = self.encoder.embeddings.word_embeddings(input_ids)
        text_dict['attention_mask'] = attention_mask
        text_dict['token_type_ids'] = token_type_ids
        text_dict['position_ids'] = self.encoder.embeddings.position_ids[:, :input_ids.size(1)]

        neighborhood_dict = {}
        outputs_embeds, attention_mask, token_type_ids, position_ids = self.neighborhood_Embedding(
            # self.E_emb.data, 
            self.E_emb, 
            h_idx, 
            r_idx=None, 
            sample_neighborhood=sample_neighborhood, 
            sample_neighborhood_mask=sample_neighborhood_mask, 
            mask_idx=mask_idx)
        neighborhood_dict['inputs_embeds'] = outputs_embeds
        neighborhood_dict['attention_mask'] = attention_mask
        neighborhood_dict['token_type_ids'] = token_type_ids
        neighborhood_dict['position_ids'] = position_ids

        if self.method == 'concat':
            output_vector = self.concat_encoding(text_dict, neighborhood_dict, r_idx)
        elif  self.method == 'linear':
            output_vector = self.linear_encoding(text_dict, neighborhood_dict, r_idx)
        elif  self.method == 'MLP':
            output_vector = self.MLP_encoding(text_dict, neighborhood_dict, r_idx)
        else:
            raise ValueError('')
        
        hw_multi_feature, hw_single_feature =  _get_sememe_feature(hw_model,hw_input_ids, hw_attention_mask)
        
        #hr_all_combined_output = torch.cat((output_vector, hw_multi_feature), dim=1)
        #hr_main_combined_output = torch.cat((output_vector, hw_single_feature), dim=1)
        #print(hr_combined_output.size())
        
        final_vector = self.dualSememeFusion(output_vector,hw_single_feature,hw_multi_feature)



        #print(final_vector.size())
        
        return  final_vector







class EMA(torch.nn.Module):
    """
    [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage]
    """

    def __init__(self, model1, model2, decay=0.75, total_step=15000):
        super().__init__()
        self.decay = decay
        self.total_step = total_step
        self.step = 0
        self.model1 = model1
        self.model2 = model2

    def update(self):
        self.step = self.step+1
        decay_new = 1-(1-self.decay)*(math.cos(math.pi*self.step/self.total_step)+1)/2
        with torch.no_grad():
            m_std = self.model1.state_dict().values()
            e_std = self.model2.state_dict().values()
            for e, m in zip(e_std, m_std):
                e.copy_(decay_new * e + (1. - decay_new) * m)


class MoCoKGC(nn.Module):

    def __init__(self, args, entities_size, relations_size, NeighborsRepresentationSets=None, num_training_steps=None):
        super().__init__()
        self.args = args
        self.entities_size = entities_size
        self.relations_size = relations_size

        self.hr_encoder:MoCo_encoder = MoCo_encoder(args, entities_size, relations_size, NeighborsRepresentationSets, method=args.entity_embedding_method, hr_encoder=True,mode='hr')
        self.e_encoder:MoCo_encoder = MoCo_encoder(args, entities_size, relations_size, NeighborsRepresentationSets, method=args.entity_embedding_method,mode='t')
        self.momentum_e_encoder:MoCo_encoder = MoCo_encoder(args, entities_size, relations_size, NeighborsRepresentationSets, method=args.entity_embedding_method,mode='t')
        self.momentum_e_encoder.load_state_dict(self.e_encoder.state_dict(), strict=True)

        if num_training_steps is not None:
            self.ema_t = EMA(self.e_encoder, self.momentum_e_encoder, decay=args.m_decay, total_step=num_training_steps)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.tau).log(), requires_grad=args.finetune_t)

    def set_E_emb(self, state_dict=None):
        if state_dict is not None and state_dict.get("E_emb", None) is not None:
            self.E_emb = torch.nn.Parameter(state_dict.pop("E_emb"), requires_grad=False)
        else:
            self.E_emb = torch.nn.Parameter(torch.empty(self.entities_size+1, self.hr_encoder.hidden_size), requires_grad=False)
            xavier_normal_(self.E_emb.data)
        self.hr_encoder.set_E_emb(self.E_emb.data)
        self.e_encoder.set_E_emb(self.E_emb.data)
        self.momentum_e_encoder.set_E_emb(self.E_emb.data)

    def encode_hr(self, hw_model,input_ids, attention_mask, token_type_ids,hw_input_ids, hw_attention_mask, hw_token_type_ids, h_idx=None, r_idx=None, sample_neighborhood=None, sample_neighborhood_mask=None, mask_idx=None):
        output_vector = self.hr_encoder(hw_model, input_ids, attention_mask, token_type_ids,hw_input_ids, hw_attention_mask, hw_token_type_ids, h_idx=h_idx, r_idx=r_idx, sample_neighborhood=sample_neighborhood, sample_neighborhood_mask=sample_neighborhood_mask, mask_idx=mask_idx)
        output_vector = F.normalize(output_vector, dim=-1)
        return output_vector
    
    def encode_t(self, hw_model,input_ids, attention_mask, token_type_ids,hw_input_ids, hw_attention_mask, hw_token_type_ids, e_idx=None, sample_neighborhood=None, sample_neighborhood_mask=None, mask_idx=None):
        output_vector = self.e_encoder(hw_model, input_ids, attention_mask, token_type_ids,hw_input_ids, hw_attention_mask, hw_token_type_ids, h_idx=e_idx, r_idx=None, sample_neighborhood=sample_neighborhood, sample_neighborhood_mask=sample_neighborhood_mask,mask_idx=mask_idx)
        output_vector = F.normalize(output_vector, dim=-1)
        return output_vector
    
    def encode_e(self, hw_model, input_ids, attention_mask, token_type_ids, hw_input_ids, hw_attention_mask, hw_token_type_ids, e_idx=None, sample_neighborhood=None, sample_neighborhood_mask=None, mask_idx=None):
        output_vector = self.momentum_e_encoder(hw_model, input_ids, attention_mask, token_type_ids,hw_input_ids, hw_attention_mask, hw_token_type_ids, h_idx=e_idx, r_idx=None, sample_neighborhood=sample_neighborhood, sample_neighborhood_mask=sample_neighborhood_mask, mask_idx=mask_idx)
        output_vector = F.normalize(output_vector, dim=-1)
        return output_vector

    @torch.no_grad()
    def update_E_enc(self,hw_model, entity_loader):
        loader = tqdm.tqdm(entity_loader)
        for batch_data in loader:
            batch_data = data_to_device(batch_data, self.args.device)
            e_idx = batch_data['entity_inputs']['e_idx']
            e = self.encode_e(hw_model,**batch_data['entity_inputs'])
            self.E_emb.data[e_idx, :] = e.to(self.E_emb.data.device)
        self.E_emb.requires_grad = False

    def forward(self, hw_model,input_ids, attention_mask, token_type_ids, hw_input_ids, hw_attention_mask, hw_token_type_ids,  h_idx=None, r_idx=None, sample_neighborhood=None, sample_neighborhood_mask=None):
        Enc_hr = self.encode_hr(hw_model,input_ids, attention_mask, token_type_ids,hw_input_ids, hw_attention_mask, hw_token_type_ids, h_idx, r_idx, sample_neighborhood=sample_neighborhood, sample_neighborhood_mask=sample_neighborhood_mask)
        score = get_sim(Enc_hr.to(self.E_emb.data.device), self.E_emb.data)
        return score
            

def last_hidden_state_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
    output_vector = sum_embeddings / sum_mask
    return output_vector
