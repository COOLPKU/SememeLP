import numpy as np
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW
from MoCoKGC import MoCoKGC
from utils import negative_sampling_Entities, Entities, Konwledge_Graph, data_to_device, batch_split, get_sim, get_logger, set_seed, collate
import tqdm
import time
from evaluator import Evaluator
import random
import copy
from typing import List
from drqueue import Duplicate_Removed_Queue

tiny_batch_rate = 1
iterate_tail = 0
class Trainer(object):

    def __init__(self, args, model,hw_model, evaluator, entities, train_data):
        print("Start")
        self.args = args
        self.logger = get_logger(args.log_file)
        self.model:MoCoKGC = model
        self.hw_model = hw_model
        self.evaluator:Evaluator = evaluator
        self.entities:Entities = entities
        self.entities_loader = torch.utils.data.DataLoader(self.entities,
                            batch_size=self.args.test_batch_size*tiny_batch_rate,
                            shuffle=False,
                            collate_fn=collate,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)
        self.negative_sampling_entities:Entities = negative_sampling_Entities(len(train_data)//self.args.batch_size * args.extra_negative_sample_size, entities)
        self.negative_sampling_loader = torch.utils.data.DataLoader(self.negative_sampling_entities,
                            # sampler=sampler,
                            shuffle=False,
                            batch_size=args.extra_negative_sample_size,
                            collate_fn=collate,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=True)
        self.train_data:Konwledge_Graph = train_data
        self.train_loader = torch.utils.data.DataLoader(self.train_data,
                            batch_size=self.args.batch_size,
                            shuffle=True,
                            collate_fn=collate,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=True)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'lr': self.args.lr, 'betas':(0.9,0.999), 'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'lr': self.args.lr, 'betas':(0.9,0.999), 'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        num_training_steps = args.epochs * len(self.train_data) // max(args.batch_size, 1)
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        self.CE_criterion = torch.nn.CrossEntropyLoss()
        self.scaler = GradScaler(enabled = self.args.AMP_enabled)
        self.step = 0
        self.queue_size = args.queue_size

        if self.queue_size>0:
            self.queue_sample_idx = Duplicate_Removed_Queue(len(entities))

    def pretrain(self):
        self.logger.info(f"data_dir: {self.args.data_dir}")
        self.logger.info(f"pretraining...")
        entities_loader = torch.utils.data.DataLoader(self.entities,
                            batch_size=self.args.batch_size,
                            shuffle=True,
                            collate_fn=collate,
                            num_workers=self.args.num_workers,
                            pin_memory=False,
                            drop_last=True)
        for n, p in self.model.hr_bert.named_parameters():
            if 'bert' in n:
                p.requires_grad = False 
            else:
                p.requires_grad = True
        for epoch in range(1, self.args.epochs):
            self.logger.info(f"epoch:{epoch}")
            loss = []   
            accuracies = []
            self.model.train() 
            loader = tqdm.tqdm(entities_loader)
            for batch_data in loader:
                batch_data = data_to_device(batch_data, self.args.device)
                with autocast(enabled = self.args.AMP_enabled): 
                    batch_loss, accuracy = self.pretrin_iterate(batch_data)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                loss.append(batch_loss)
                accuracies.append(accuracy)
                loader.set_description_str(f'epoch:{epoch}, loss:{np.mean(loss):.6}, accuracy:{np.mean(accuracies):.6}')
            self.logger.info(f"epoch:{epoch}, loss:{np.mean(loss):.6}, accuracy:{np.mean(accuracies):.6}")
            torch.save(self.model.state_dict(),self.args.output_path+'best.pt')

    def pretrin_iterate(self, batch_data):
        batch_data["entity_inputs"]["h_idx"] = batch_data["e_idx"]
        hr = self.model.encode_hr(**batch_data["entity_inputs"])
        pred = self.model.classification(hr)
        accuracy = (torch.argmax(pred, dim=-1)==batch_data['e_idx']).float().mean().item()
        loss = self.CE_criterion(pred, batch_data['e_idx'])
        self.scaler.scale(loss).backward() 
        return loss.item(), accuracy

    def train_and_valid(self, iterate_function):
        best_metric = 0
        Wikidata5M_tag =  self.args.data_dir=="./data/Wikidata5M/wikidata5m_inductive/" or self.args.data_dir=="./data/Wikidata5M/wikidata5m_transductive/"
        updata_E_enc_frequency = 200 if not Wikidata5M_tag else 4000
        
        for epoch in range(1, self.args.epochs+1):
            self.logger.info(f"epoch:{epoch}")
            loss = []
            acc = []
            if self.args.checkpoint_path is None:
                print("初始化实体")
                self.model.update_E_enc(self.hw_model,self.entities_loader)
                torch.save(self.model.state_dict(),self.args.output_path+'init_E_enc.pt')
            if self.queue_size>0:
                self.queue_sample_idx.working(self.queue_size, torch.arange(len(self.entities)-self.queue_size, len(self.entities)).tolist())

            self.model.train() 
            loader = tqdm.tqdm(self.train_loader)
            for i, (batch_data, batch_negative_samples) in enumerate(zip(loader, self.negative_sampling_loader)):

                batch_data = data_to_device(batch_data, self.args.device)
                batch_negative_samples = data_to_device(batch_negative_samples, self.args.device)
                
                with autocast(enabled = self.args.AMP_enabled): 
                    batch_loss, batch_acc = iterate_function(batch_data, batch_negative_samples)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.model.ema_t.update() #EMA
                loss.append(batch_loss)
                acc.append(batch_acc)
                loader.set_description_str(f'epoch:{epoch}, loss:{np.mean(loss):.6}, acc:{np.mean(acc, axis=0)}')
                if i!=0 and i%updata_E_enc_frequency==0:
                    self.logger.info(f"{i}个step评价")
                    self.logger.info(f"epoch:{epoch}, loss:{np.mean(loss):.6}, acc:{np.mean(acc, axis=0)}")
                    self.model.eval() 
                    metric = self.evaluator.valid(self.model,self.hw_model, updata_E_enc=True, inductive=True if self.args.data_dir=="./data/Wikidata5M/wikidata5m_inductive/" else False)
                    self.logger.info(f"forward_metric :{metric['forward_metric']}")
                    self.logger.info(f"backward_metric:{metric['backward_metric']}")
                    self.logger.info(f"mean_metric    :{metric['mean_metric']}")
                    mrr = metric['mean_metric']['MRR']
                    if mrr > best_metric:
                        best_metric = mrr
                        torch.save(self.model.state_dict(),self.args.output_path+'best.pt')
                    self.model.train() 

            self.logger.info(f"epoch:{epoch}, loss:{np.mean(loss):.6}, acc:{np.mean(acc, axis=0)}")
            self.model.eval() 

            if not Wikidata5M_tag:
                metric = self.evaluator.train(self.model, self.hw_model,updata_E_enc=True) ############################
                self.logger.info(f"forward_metric :{metric['forward_metric']}")
                self.logger.info(f"backward_metric:{metric['backward_metric']}")
                self.logger.info(f"mean_metric    :{metric['mean_metric']}")
                metric = self.evaluator.valid(self.model,self.hw_model, updata_E_enc=False)
            else:
                metric = self.evaluator.valid(self.model,self.hw_model,updata_E_enc=True, inductive=True if self.args.data_dir=="./data/Wikidata5M/wikidata5m_inductive/" else False)
            self.logger.info(f"forward_metric :{metric['forward_metric']}")
            self.logger.info(f"backward_metric:{metric['backward_metric']}")
            self.logger.info(f"mean_metric    :{metric['mean_metric']}")
            mrr = metric['mean_metric']['MRR']

            metric = self.evaluator.test(self.model, self.hw_model,updata_E_enc=False, inductive=True if self.args.data_dir=="./data/Wikidata5M/wikidata5m_inductive/" else False)
            self.logger.info(f"forward_metric :{metric['forward_metric']}")
            self.logger.info(f"backward_metric:{metric['backward_metric']}")
            self.logger.info(f"mean_metric    :{metric['mean_metric']}")

            torch.save(self.model.state_dict(),self.args.output_path+'last.pt')
            if mrr > best_metric:
                best_metric = mrr
                torch.save(self.model.state_dict(),self.args.output_path+'best.pt')
        
        self.logger.info(f"testing...")
        checkpoint_path = self.args.output_path + 'best.pt'
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.args.device)
        self.model.E_emb.data = self.model.E_emb.data.to("cpu")

        self.model.eval() 
        metric = self.evaluator.test(self.model,self.hw_model, updata_E_enc=False, inductive=True if self.args.data_dir=="./data/Wikidata5M/wikidata5m_inductive/" else False)
        self.logger.info(f"forward_metric :{metric['forward_metric']}")
        self.logger.info(f"backward_metric:{metric['backward_metric']}")
        self.logger.info(f"mean_metric    :{metric['mean_metric']}")

    def run(self):
        self.logger.info(f"data_dir: {self.args.data_dir}")
        self.logger.info(f"training...")
        self.train_and_valid(self.cumulative_iterate)

    def iterate(self, batch_data):
        raise EOFError
        batch_size = len(batch_data['h_idx'])
        targets = batch_data["targets"][:,batch_data["t_idx"]]
        triple_mask = (targets>0).fill_diagonal_(False)
        label = torch.arange(batch_size).to(self.args.device)
        margin = torch.zeros((batch_size, batch_size)).fill_diagonal_(self.args.margin).to(self.args.device)
        if self.args.add_relation_prompt:
            batch_data["hr_inputs"]["r_idx"] = batch_data["r_idx"]
        hr = self.model.encode_hr(**batch_data["hr_inputs"])
        t = self.model.encode_t(**batch_data["t_inputs"])
        score = (get_sim(hr, t) - margin)*self.model.log_inv_t.exp()
        score.masked_fill_(triple_mask, -1e4)
        loss = self.CE_criterion(score, label)
        loss += self.CE_criterion(score.T, label)
        self.scaler.scale(loss).backward() 
        return loss.item()

    def cumulative_iterate(self, batch_data, batch_negative_samples):

        def get_loss_acc_and_backward(hr, E_enc, margins=None, triple_mask=None, labels=None, other_loss=0, only_get_loss=False, loss_weight = 1):
            if margins is not None:
                score = get_sim(hr, E_enc) - margins
            else:
                score = get_sim(hr, E_enc)

            score = self.model.log_inv_t.exp() * score

            if triple_mask is not None:
                score.masked_fill_(triple_mask, -1e4)

            loss = self.CE_criterion(score, labels)
            loss = loss_weight*loss + other_loss
            if only_get_loss:
                return loss
            
            acc = torch.sum(torch.argmax(score.detach(), dim=-1)==labels)/labels.size(0)
            self.scaler.scale(loss).backward()
            return loss.item(), acc.item(), 0
        
        def get_mask(hr_targets:List[torch.Tensor], idx:torch.Tensor):
            mask = []
            for i, hr_target in enumerate(hr_targets):
                search_index = torch.searchsorted(hr_target, idx)
                search_index.clamp_max_(hr_target.size(0)-1)
                mask_i = hr_target[search_index]==idx
                mask.append(mask_i)
            mask = torch.stack(mask, dim=0)
            return mask


        
        batch_size = len(batch_data["hr_inputs"]['h_idx'])
        hr_targets = batch_data.pop("targets")
        data_list = batch_split(batch_data, batch_size, self.args.actual_batch_size)
        negative_samples_list = batch_split(batch_negative_samples, len(batch_negative_samples["entity_inputs"]["e_idx"]), self.args.actual_batch_size)
        h_idx = batch_data["hr_inputs"]['h_idx']
        t_idx = batch_data["t_inputs"]["e_idx"]
        e_idx = batch_negative_samples["entity_inputs"]["e_idx"]
        batch_acc = []
        batch_acc_ = []
        batch_loss = []
        seed = random.randint(0,2**20)

        # step1
        for param in self.model.parameters():
            param.requires_grad = False

        set_seed(seed)
        hr = []
        for i,data in enumerate(data_list):
            hr_i = self.model.encode_hr(self.hw_model, **data["hr_inputs"])
            hr.append(hr_i)
        hr = torch.concat(hr, dim=0).detach()

        set_seed(seed)
        t = []
        for i,data in enumerate(data_list):
            t_i = self.model.encode_t(self.hw_model, **data["t_inputs"])
            t.append(t_i)
        t = torch.concat(t, dim=0).detach()

        set_seed(seed)
        e = []
        for i,data in enumerate(negative_samples_list):
            e_i = self.model.encode_t(self.hw_model, **data["entity_inputs"])
            e.append(e_i)
        e = torch.concat(e, dim=0).detach()

        set_seed(seed)
        pt = []
        for i,data in enumerate(data_list):
            pt_i = self.model.encode_e(self.hw_model, **data["t_inputs"])
            pt.append(pt_i)
        pt = torch.concat(pt, dim=0).detach()

        set_seed(seed)
        pe = []
        for i,data in enumerate(negative_samples_list):
            pe_i = self.model.encode_e(self.hw_model, **data["entity_inputs"])
            pe.append(pe_i)
        pe = torch.concat(pe, dim=0).detach()


        for param in self.model.parameters():
            param.requires_grad = True


        labels = torch.arange(batch_size).to(self.args.device)
        sample_idx = torch.concat((t_idx, e_idx), dim=0)
        sample = torch.concat((t, e), dim=0)
        
        if self.queue_size>0:
            queue_sample_idx_list = self.queue_sample_idx.get_queue_list()
        else:
            queue_sample_idx_list = []
        _queue_sample_idx_list_plus = queue_sample_idx_list + sample_idx.to('cpu').tolist()
        queue_sample_idx_list_plus = torch.tensor(_queue_sample_idx_list_plus).to(labels.device)

        margins_t = torch.zeros((t_idx.size(0), sample_idx.size(0))).fill_diagonal_(self.args.margin).to(self.args.device)
        triple_mask_t = get_mask(hr_targets=hr_targets, idx=sample_idx).fill_diagonal_(False)

        queue_margins_t = F.one_hot(len(queue_sample_idx_list) + labels, len(queue_sample_idx_list_plus)) * self.args.margin
        queue_triple_mask_t = get_mask(hr_targets=hr_targets, idx=queue_sample_idx_list_plus).masked_fill_(queue_margins_t>0, False)

        p_sample = torch.concat((pt, pe), dim=0)
        self.model.E_emb.data[sample_idx,:] = p_sample.to(self.model.E_emb.data.device)


        # step2
        self.optimizer.zero_grad()
        E_enc = self.model.E_emb.data[:-1]
        E_enc_sample = E_enc[_queue_sample_idx_list_plus].to(labels.device)
        set_seed(seed)
        for i, data in enumerate(data_list):
            i_slice = slice(i*self.args.actual_batch_size, (i+1)*self.args.actual_batch_size)
            hr_i = self.model.encode_hr(self.hw_model,**data["hr_inputs"])
            hr[i_slice, :] = hr_i
            
            other_loss = get_loss_acc_and_backward(
                hr, 
                sample, 
                margins_t, 
                triple_mask_t, 
                labels, 
                other_loss=0,
                only_get_loss=True
            )
            
            loss, acc, acc_ = get_loss_acc_and_backward(
                hr,
                E_enc_sample, 
                queue_margins_t, 
                queue_triple_mask_t, 
                len(queue_sample_idx_list) + labels, 
                other_loss=other_loss
            )
            
            hr = hr.detach()
            batch_loss.append(loss)
            batch_acc.append(acc)
            batch_acc_.append(acc_)

        set_seed(seed)
        for i, data in enumerate(data_list):
            i_slice = slice(i*self.args.actual_batch_size,(i+1)*self.args.actual_batch_size)
            t_i = self.model.encode_t(self.hw_model, **data["t_inputs"])
            sample[i_slice,:] = t_i

            loss, acc, acc_ = get_loss_acc_and_backward(
                hr, 
                sample, 
                margins_t, 
                triple_mask_t, 
                labels, 
                other_loss=0,
            )
            sample = sample.detach()

        set_seed(seed)
        for i, data in enumerate(negative_samples_list):
            i_slice = slice(t_idx.size(0) + i*self.args.actual_batch_size, t_idx.size(0) + (i+1)*self.args.actual_batch_size)
            e_i = self.model.encode_t(self.hw_model, **data["entity_inputs"])
            sample[i_slice,:] = e_i

            loss, acc, acc_ = get_loss_acc_and_backward(
                hr, 
                sample, 
                margins_t, 
                triple_mask_t, 
                labels, 
                other_loss=0,
            )
            sample = sample.detach()

        if self.queue_size>0:
            self.queue_sample_idx.working(self.queue_size, sample_idx.to('cpu').tolist())
        return np.mean(batch_loss), (np.mean(batch_acc), np.mean(batch_acc_),)

