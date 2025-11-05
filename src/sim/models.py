from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

from triplet_mask import construct_mask

def _get_sememe_feature(hw_model,token_ids, mask):
    hw_model.eval()
    with torch.no_grad():
        cls_a_hidden, cls_b_hidden, multi_logits, single_logits = hw_model.predict(token_ids, mask)
    return cls_a_hidden.detach(),cls_b_hidden.detach()

class DualSememeFusion(nn.Module):
    def __init__(self, in_dim=768, out_dim=768):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim
        
        

        # === IGF ===
        self.gate_main = self._build_gate()
        self.gate_all = self._build_gate()

        # === WF ===
        self.balance_layer = nn.Sequential(
            nn.Linear(2*out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, 2),
            nn.Softmax(dim=-1)
        )

        # === FGF ===
        self.joint_fuse = nn.Sequential(
            nn.Linear(4*out_dim, 8*out_dim),  # 保持原始扩展比例
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(8*out_dim, out_dim),    # 维持原始压缩比例
            nn.Sigmoid()
        )

        self._init_weights()

    def _build_gate(self):
        return nn.Sequential(
            nn.Linear(2*self.out_dim, 4*self.out_dim),
            nn.GELU(),
            nn.Linear(4*self.out_dim, self.out_dim),
            nn.Sigmoid()
        )

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, h_text, h_main, h_all):
       

        # IGF
        gate_main = self.gate_main(torch.cat([h_text, h_main], -1))
        fuse_main = gate_main * h_text + (1 - gate_main) * h_main
        
        gate_all = self.gate_all(torch.cat([h_text, h_all], -1))
        fuse_all = gate_all * h_text + (1 - gate_all) * h_all

        # WF
        balance_weights = self.balance_layer(torch.cat([fuse_main, fuse_all], -1))
        balanced = balance_weights[:, 0:1] * fuse_main + balance_weights[:, 1:2] * fuse_all

        # FGF
        joint_input = torch.cat([h_text, balanced, fuse_main, fuse_all], -1)
        final_gate = self.joint_fuse(joint_input)
        
        return final_gate * h_text + (1 - final_gate) * balanced





def build_model(args) -> nn.Module:
    return CustomBertModel(args)





@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


    


import torch
import torch.nn as nn
import torch.nn.functional as F





    


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, 768)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)
        
        
        self.hr_dual = DualSememeFusion()
        self.tail_dual = DualSememeFusion()
        

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, hw_model, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                tail_hw_token_ids,tail_hw_mask,
                head_hw_token_ids,head_hw_mask,
                
                
                
                only_ent_embedding=False, **kwargs) -> dict:
        
        head_hw_multi_feature, head_hw_single_feature =  _get_sememe_feature(hw_model,head_hw_token_ids, head_hw_mask)
        tail_hw_multi_feature, tail_hw_single_feature =  _get_sememe_feature(hw_model,tail_hw_token_ids, tail_hw_mask)
        
        if only_ent_embedding:
            with torch.no_grad():
                tail_feature = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
                tail_vector = self.tail_dual(tail_feature,tail_hw_single_feature,tail_hw_multi_feature)
                tail_vector = torch.nn.functional.normalize(tail_vector, p=2, dim=1)
                return {'ent_vectors': tail_vector.detach()}
            
        
        
        
        
        hr_feature = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)
     
        
        hr_vector = self.hr_dual(hr_feature,head_hw_single_feature,head_hw_multi_feature)
        hr_vector = torch.nn.functional.normalize(hr_vector, p=2, dim=1)
       
        
        

        tail_feature = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        tail_vector = self.tail_dual(tail_feature,tail_hw_single_feature,tail_hw_multi_feature)
        tail_vector = torch.nn.functional.normalize(tail_vector, p=2, dim=1)
        
        
        
        
        head_feature = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)
        

        head_vector = self.tail_dual(head_feature,head_hw_single_feature,head_hw_multi_feature)

        head_vector = torch.nn.functional.normalize(head_vector, p=2, dim=1)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    #output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
