import torch
import numpy as np
import tqdm
from utils import Entities, collate, data_to_device
from MoCoKGC import MoCoKGC
import pickle
import json
class Evaluator(object):
    def __init__(self, args, entities, valid_data, valid_reverse_data, test_data, test_reverse_data, test_train_data=None, test_train_reverse_data=None):
        self.args = args
        self.entities:Entities = entities
        self.entities_loader = self.get_loader(entities, self.args.actual_batch_size)
        self.valid_data = valid_data
        self.valid_loader = self.get_loader(valid_data, self.args.test_batch_size)
        self.valid_reverse_data = valid_reverse_data
        self.valid_reverse_loader = self.get_loader(valid_reverse_data, self.args.test_batch_size)
        self.test_data = test_data
        self.test_loader = self.get_loader(test_data, self.args.test_batch_size)
        self.test_reverse_data = test_reverse_data
        self.test_reverse_loader = self.get_loader(test_reverse_data, self.args.test_batch_size)
        if test_train_data is not None:
            self.train_data = test_train_data
            self.train_loader = self.get_loader(test_train_data)
            self.train_reverse_data = test_train_reverse_data
            self.train_reverse_loader = self.get_loader(test_train_reverse_data)

    def get_loader(self, data, shuffle=False, batch_size=None):
        return torch.utils.data.DataLoader(data,
                            batch_size=self.args.test_batch_size if batch_size is None else batch_size,
                            shuffle=False,
                            collate_fn=collate,
                            num_workers=self.args.num_workers,
                            pin_memory=True,
                            drop_last=False)

    @torch.no_grad()
    def evalue(self, loader, model:MoCoKGC,hw_model, inductive=False,name='未知.json'):
        hits = []
        ranks = []
        test_info=[]
        r_idx_r = []
        for i in range(10):
            hits.append([])

        if inductive:
            inductive_entity = set()
            for triple in loader.dataset.triples:
                inductive_entity.add(triple[0].item())
                inductive_entity.add(triple[2].item())
            inductive_entity = list(inductive_entity)

            

        loader = tqdm.tqdm(loader)
        for batch_data in loader:
            data = data_to_device(batch_data, self.args.device)
            h_idx, r_idx, t_idx = data["hr_inputs"]['h_idx'], data["hr_inputs"]['r_idx'], data["hr_inputs"]['mask_idx']
            r_idx_r.extend(r_idx.to("cpu").tolist())
            data["hr_inputs"].pop('mask_idx')
            predictions = model(hw_model,**data["hr_inputs"])

            for j in range(h_idx.size(0)):
                # filt = self.valid_data.all_hr_vocab[(h_idx[j].item(), r_idx[j].item())]
                filt = self.valid_data.train_serialized_hr_vocab[(h_idx[j].item(), r_idx[j].item())].tolist()
                filt.extend(list(self.valid_data.test_serialized_hr_vocab.get((h_idx[j].item(), r_idx[j].item()), {})))
                target_value = predictions[j,t_idx[j].item()].item()
                predictions[j, filt] = -torch.inf
                predictions[j, t_idx[j].item()] = target_value

            if inductive:
                predictions = predictions[:,inductive_entity]
            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(h_idx.size(0)):
                sort_idxs_j = sort_idxs[j] 
                if inductive:
                    for ji in range(len(sort_idxs_j)):
                        sort_idxs[j][ji] = inductive_entity[sort_idxs[j][ji]]
                rank = np.where(sort_idxs_j==t_idx[j].item())[0][0]
                ranks.append(rank+1)
                test_info.append({'tripler':(int(h_idx[j].item()), int(r_idx[j].item()), int(t_idx[j].item())),'rank':int(rank),'predictions':sort_idxs_j[0:11].tolist()})
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
        metric = {
            'MRR': np.mean(1./np.array(ranks)),
            'MR': np.mean(ranks),
            'Hits1': np.mean(hits[0]),
            'Hits3': np.mean(hits[2]),
            'Hits10': np.mean(hits[9]),
            
        }

        with open(f'./ana_1/{name}_test.jsonl', 'w', encoding='utf-8') as f:
        # 遍历JSON数据的每个键值对
            for line in test_info:
                # 写入TXT文件，键和值之间用制表符分隔
                f.write(json.dumps(line, ensure_ascii=False).strip() + "\n")
 

            
        
        return metric, (ranks, r_idx_r)

    def valid(self, model:MoCoKGC,hw_model, updata_E_enc=True, inductive=False):
        if updata_E_enc:
            model.update_E_enc(hw_model,self.entities_loader)
        forward_metric, forward_ranks = self.evalue(self.valid_loader, model,hw_model, inductive=inductive)
        backward_metric, backward_ranks = self.evalue(self.valid_reverse_loader, model,hw_model, inductive=inductive)
        metric = {
        "forward_metric":forward_metric,
        "backward_metric":backward_metric,
        "mean_metric":{k:(backward_metric[k]+v)/2 for k,v in forward_metric.items()}
        }
        return metric

    def test(self, model:MoCoKGC,hw_model, updata_E_enc=True, inductive=False):
        if updata_E_enc:
            model.update_E_enc(hw_model,self.entities_loader)
        forward_metric, forward_ranks = self.evalue(self.test_loader, model,hw_model, inductive=inductive,name='forward')
        backward_metric, backward_ranks = self.evalue(self.test_reverse_loader, model,hw_model, inductive=inductive,name='backward')
        metric = {
        "forward_metric":forward_metric,
        "backward_metric":backward_metric,
        "mean_metric":{k:(backward_metric[k]+v)/2 for k,v in forward_metric.items()}
        }
        return metric


    @torch.no_grad()
    def evalue_sample(self, loader, model:MoCoKGC,hw_model, n):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        loader = tqdm.tqdm(loader)
        for i, batch_data in enumerate(loader):
            if i>n:
                break
            data = data_to_device(batch_data, self.args.device)
            h_idx, r_idx, t_idx = data["hr_inputs"]['h_idx'], data["hr_inputs"]['r_idx'], data["hr_inputs"]['mask_idx']
            data["hr_inputs"].pop('mask_idx')
            predictions = model(hw_model,**data["hr_inputs"])

            for j in range(h_idx.size(0)):
                # filt = self.valid_data.all_hr_vocab[(h_idx[j].item(), r_idx[j].item())]
                filt = self.valid_data.train_serialized_hr_vocab[(h_idx[j].item(), r_idx[j].item())].tolist()
                filt.extend(list(self.valid_data.test_serialized_hr_vocab.get((h_idx[j].item(), r_idx[j].item()), {})))
                
                target_value = predictions[j,t_idx[j].item()].item()
                predictions[j, filt] = -torch.inf
                predictions[j, t_idx[j].item()] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(h_idx.size(0)):
                rank = np.where(sort_idxs[j]==t_idx[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        metric = {
            'MRR': np.mean(1./np.array(ranks)),
            'MR': np.mean(ranks),
            'Hits1': np.mean(hits[0]),
            'Hits3': np.mean(hits[2]),
            'Hits10': np.mean(hits[9]),
            
        }

        


        
        
        return metric
    
    def train(self, model:MoCoKGC,hw_model, updata_E_enc=True):
        if updata_E_enc:
            model.update_E_enc(hw_model,self.entities_loader)
        forward_metric = self.evalue_sample(self.train_loader, model,hw_model, 20)
        backward_metric = self.evalue_sample(self.train_reverse_loader, model, hw_model,20)
        metric = {
        "forward_metric":forward_metric,
        "backward_metric":backward_metric,
        "mean_metric":{k:(backward_metric[k]+v)/2 for k,v in forward_metric.items()}
        }
        return metric

    @torch.no_grad()
    def evalue_(self, loader, model:MoCoKGC):
        hits = [[],[],[]]
        ranks = [[],[],[]]
        for i in range(10):
            hits[0].append([])
            hits[1].append([])
            hits[2].append([])

        loader = tqdm.tqdm(loader)
        for batch_data in loader:
            data = data_to_device(batch_data, self.args.device)
            h_idx, r_idx, t_idx = data['h_idx'], data['r_idx'], data['t_idx']
            if self.args.add_relation_prompt:
                data["hr_inputs"]["r_idx"] = r_idx
            if self.args.add_entity_prompt:
                data["hr_inputs"]["h_idx"] = data["h_idx"]

            pred_Enc, pred_Emb, pred = model(**data["hr_inputs"])

            for pi, predictions in enumerate([pred_Enc, pred_Emb, pred]):
                for j in range(h_idx.size(0)):
                    # filt = self.valid_data.all_hr_vocab[(h_idx[j].item(), r_idx[j].item())]
                    filt = self.valid_data.train_serialized_hr_vocab[(h_idx[j].item(), r_idx[j].item())].tolist()
                    filt.extend(list(self.valid_data.test_serialized_hr_vocab.get((h_idx[j].item(), r_idx[j].item()), {})))
                    
                    target_value = predictions[j,t_idx[j].item()].item()
                    predictions[j, filt] = -torch.inf
                    predictions[j, t_idx[j].item()] = target_value

                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

                sort_idxs = sort_idxs.cpu().numpy()
                for j in range(h_idx.size(0)):
                    rank = np.where(sort_idxs[j]==t_idx[j].item())[0][0]
                    ranks[pi].append(rank+1)

                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[pi][hits_level].append(1.0)
                        else:
                            hits[pi][hits_level].append(0.0)

        metric = [{
            'MRR': np.mean(1./np.array(ranks[pi])),
            'MR': np.mean(ranks[pi]),
            'Hits1': np.mean(hits[pi][0]),
            'Hits3': np.mean(hits[pi][2]),
            'Hits10': np.mean(hits[pi][9]),
        } for pi in range(3)]
        return metric

    def valid_(self, model:MoCoKGC):
        model.update_E_enc(self.entities_loader)
        forward_metric = self.evalue_(self.valid_loader, model)
        backward_metric = self.evalue_(self.valid_reverse_loader, model)
        metric = {
        "forward_metric":forward_metric,
        "backward_metric":backward_metric,
        "mean_metric":[{k:(backward_metric[pi][k]+v)/2 for k,v in forward_metric[pi].items()} for pi in range(3)]
        }
        return metric

    def test_(self, model:MoCoKGC):
        model.update_E_enc(self.entities_loader)
        forward_metric = self.evalue_(self.test_loader, model)
        backward_metric = self.evalue_(self.test_reverse_loader, model)
        metric = {
        "forward_metric":forward_metric,
        "backward_metric":backward_metric,
        "mean_metric":[{k:(backward_metric[pi][k]+v)/2 for k,v in forward_metric[pi].items()} for pi in range(3)]
        }
        return metric


    @torch.no_grad()
    def evalue_sample_(self, loader, model:MoCoKGC, n):
        hits = [[],[],[]]
        ranks = [[],[],[]]
        for i in range(10):
            hits[0].append([])
            hits[1].append([])
            hits[2].append([])

        loader = tqdm.tqdm(loader)
        for i, batch_data in enumerate(loader):
            if i>n:
                break
            data = data_to_device(batch_data, self.args.device)
            h_idx, r_idx, t_idx = data['h_idx'], data['r_idx'], data['t_idx']
            if self.args.add_relation_prompt:
                data["hr_inputs"]["r_idx"] = r_idx
            pred_Enc, pred_Emb, pred = model(**data["hr_inputs"])

            for pi, predictions in enumerate([pred_Enc, pred_Emb, pred]):
                for j in range(h_idx.size(0)):
                    # filt = self.valid_data.all_hr_vocab[(h_idx[j].item(), r_idx[j].item())]
                    filt = self.valid_data.train_serialized_hr_vocab[(h_idx[j].item(), r_idx[j].item())].tolist()
                    filt.extend(list(self.valid_data.test_serialized_hr_vocab.get((h_idx[j].item(), r_idx[j].item()), {})))
                    target_value = predictions[j,t_idx[j].item()].item()
                    predictions[j, filt] = -torch.inf
                    predictions[j, t_idx[j].item()] = target_value

                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

                sort_idxs = sort_idxs.cpu().numpy()
                for j in range(h_idx.size(0)):
                    rank = np.where(sort_idxs[j]==t_idx[j].item())[0][0]
                    ranks[pi].append(rank+1)

                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[pi][hits_level].append(1.0)
                        else:
                            hits[pi][hits_level].append(0.0)

        metric = [{
            'MRR': np.mean(1./np.array(ranks[pi])),
            'MR': np.mean(ranks[pi]),
            'Hits1': np.mean(hits[pi][0]),
            'Hits3': np.mean(hits[pi][2]),
            'Hits10': np.mean(hits[pi][9]),
        } for pi in range(3)]
        return metric

    def train_(self, model:MoCoKGC):
        model.update_E_enc(self.entities_loader)
        forward_metric = self.evalue_sample_(self.train_loader, model)
        backward_metric = self.evalue_sample_(self.train_reverse_loader, model)
        metric = {
        "forward_metric":forward_metric,
        "backward_metric":backward_metric,
        "mean_metric":[{k:(backward_metric[pi][k]+v)/2 for k,v in forward_metric[pi].items()} for pi in range(3)]
        }
        return metric

    @torch.no_grad()
    def tmp_evalue(self, loader, hr_tensor, entity_tensor):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        total = hr_tensor.size(0)
        mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0

        loader = tqdm.tqdm(loader)
        for i, batch_data in enumerate(loader):
            data = data_to_device(batch_data, self.args.device)
            h_idx, r_idx, t_idx = data['h_idx'], data['r_idx'], data['t_idx']
            predictions = hr_tensor[i*self.args.test_batch_size:(i+1)*self.args.test_batch_size,:] @ entity_tensor.T
            # predictions = model(**data["hr_inputs"])

            for j in range(h_idx.size(0)):
                # filt = self.valid_data.all_hr_vocab[(h_idx[j].item(), r_idx[j].item())]
                filt = self.valid_data.train_serialized_hr_vocab[(h_idx[j].item(), r_idx[j].item())].tolist()
                filt.extend(list(self.valid_data.test_serialized_hr_vocab.get((h_idx[j].item(), r_idx[j].item()), {})))
                mask_indices = []
                for e_id in filt:
                    if e_id == t_idx[j].item():
                        continue
                    mask_indices.append(e_id)
                mask_indices = torch.LongTensor(mask_indices).to(predictions.device)
                predictions[j].index_fill_(0, mask_indices, -1)
                # target_value = predictions[j,t_idx[j].item()].item()
                # predictions[j, filt] = -torch.inf
                # predictions[j, t_idx[j].item()] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            # sort_idxs = sort_idxs.cpu().numpy()
            # for j in range(h_idx.size(0)):
            #     rank = np.where(sort_idxs[j]==t_idx[j].item())[0][0]
            #     ranks.append(rank+1)

            #     for hits_level in range(10):
            #         if rank <= hits_level:
            #             hits[hits_level].append(1.0)
            #         else:
            #             hits[hits_level].append(0.0)

            target_rank = torch.nonzero(sort_idxs.eq(t_idx.view(-1,1).to(sort_idxs.device)).long(), as_tuple=False)
            assert target_rank.size(0) == sort_idxs.size(0)
            for idx in range(sort_idxs.size(0)):
                idx_rank = target_rank[idx].tolist()
                assert idx_rank[0] == idx
                cur_rank = idx_rank[1]

                # 0-based -> 1-based
                cur_rank += 1
                mean_rank += cur_rank
                mrr += 1.0 / cur_rank
                hit1 += 1 if cur_rank <= 1 else 0
                hit3 += 1 if cur_rank <= 3 else 0
                hit10 += 1 if cur_rank <= 10 else 0
                ranks.append(cur_rank)

        # metrics = {
        #     'MRR': np.mean(1./np.array(ranks)),
        #     'MR': np.mean(ranks),
        #     'Hits1': np.mean(hits[0]),
        #     'Hits3': np.mean(hits[2]),
        #     'Hits10': np.mean(hits[9]),
        # }
        metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
        metrics = {k: round(v / total, 4) for k, v in metrics.items()}
        return metrics

    def tmp_test(self):
        hr_tensor = torch.load("hr_tensor.pt")
        entity_tensor = torch.load("entity_tensor.pt")
        # forward_metric = self.tmp_evalue(self.test_loader, hr_tensor, entity_tensor)
        backward_metric = self.tmp_evalue(self.test_reverse_loader, hr_tensor, entity_tensor)
        forward_metric = backward_metric
        metric = {
        "forward_metric":forward_metric,
        "backward_metric":backward_metric,
        "mean_metric":{k:(backward_metric[k]+v)/2 for k,v in forward_metric.items()}
        }
        return metric

