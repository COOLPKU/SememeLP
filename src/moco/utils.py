import pandas as pd
from collections import defaultdict
import torch
from typing import Optional, List
from torch.nn import functional as F
import random
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.tokenization_utils_base import BatchEncoding
import copy
import math
import tqdm
from tensordict import TensorDict
from serialize import TorchShmSerializedList
import os
import pickle

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def get_table(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in tqdm.tqdm(lines):
        line = line[:-1].split('\t')
        data.append(line)
    return  data


class SerializedEntitiyNeighborhood(TorchShmSerializedList):
    def __init__(self, lst: List[List[List[int]]], relation_size:int):
        lst = [np.array(item, dtype=np.int64) for sublist in lst for item in sublist]
        super().__init__(lst, need_pickle=False)
        self.relation_size = relation_size
    def __getitem__(self, index):
        h = super().__getitem__(index*2)
        r = super().__getitem__(index*2+1)
        return torch.stack((h, r), dim=0)
    def __len__(self):
        return len(self._addr)//2

class SerializedHeadRelationVocab():

    def __init__(self, hr_vocab: List[dict], relation_size:int):
        self.tails_list = []
        self.hr_vocab = [defaultdict(set) for _ in range(len(hr_vocab))]
        self.relation_size = relation_size
        tails_index = 0
        for h_idx in tqdm.tqdm(range(len(hr_vocab))):
            for r_idx in hr_vocab[h_idx].keys():
                tails = hr_vocab[h_idx][r_idx]
                tails = list(tails)
                tails = sorted(tails)
                self.tails_list.append(np.array(tails, dtype=np.int64))
                self.hr_vocab[h_idx][r_idx] = tails_index
                tails_index+=1
        self.hr_vocab = TorchShmSerializedList(self.hr_vocab, need_pickle=True)
        print("hr_vocab 加载完成")
        self.tails_list = TorchShmSerializedList(self.tails_list, need_pickle=False)
        print("tails_list 加载完成")

    def __getitem__(self, hr_idx):
        h_idx, r_idx = hr_idx
        if type(h_idx) is torch.Tensor:
            h_idx = h_idx.item()
        if type(r_idx) is torch.Tensor:
            r_idx = r_idx.item()
        tails_index = self.hr_vocab[h_idx].get(r_idx, None)
        if tails_index is not None:
            tails = self.tails_list[tails_index]
        else:
            tails = torch.tensor([])
        return tails
    
    def __len__(self):
        return len(self.tails_list)
    
    def moved_to_shared_memory(self):
        self.hr_vocab.moved_to_shared_memory()
        self.tails_list.moved_to_shared_memory()
import json
class Entities(torch.utils.data.dataset.Dataset):

    def __init__(self, args, data_dir="./data/FB15k-237/", tokenizer=None, hw_tokenizer=None):
        self.args = args
        # self.entities_data = pd.read_table(data_dir+'entities_data.txt', header=None, dtype=str).values.tolist()#['Entitie','Name','Description'], dtype=np.ndarray
        self.entities_data = get_table(data_dir+'entities_data.txt')
        self.entities_data = TorchShmSerializedList(self.entities_data)
        self.entities2index = {e[0]:i for i,e in enumerate(self.entities_data)}
        with open('./ana_1/entity_dict.txt', 'w', encoding='utf-8') as f:
            json.dump(
                self.entities2index,
                f,
                indent=1,
                ensure_ascii=False,
            )

        self.neighbors = None
        self.tokenizer = tokenizer
        self.hw_tokenizer = hw_tokenizer
    
    def get_idx(self, entity):
        return self.entities2index[entity]

    def get_name(self, idx):
        return self.entities_data[idx][1]
    
    def get_description(self, idx):
        return self.entities_data[idx][2]

    def set_neighbors(self, entity_neighborhood):
        self.entity_neighborhood = entity_neighborhood

    def __len__(self):
        return len(self.entities_data)

    def __getitem__(self, index):
        description = self.get_description(index)
        # if self.neighbors is not None and len(description.split())<20:
        #     neighbors_name = [self.get_name(neighbor) for neighbor in self.neighbors.get(index, [])]
        #     description = description + "; neighbor: " + ",".join(neighbors_name)


        if self.entity_neighborhood is not None and len(description.split())<20:
            neighbors_name = [self.get_name(neighbor) for neighbor in set(self.entity_neighborhood[index][0][:30].tolist())]
            description = description + "; neighbor: " + ",".join(neighbors_name)
        entity_dict = {
            "e_idx":index,
            "name":self.get_name(index),
            "description":description,
            "ori_desc": f"{self.get_name(index)}，{self.get_description(index)}"
        }
        input_dict = self.vectorize(entity_dict)
        del entity_dict
        return input_dict

    def vectorize(self, entity_dict):
        input_dict = TensorDict({}, batch_size=[1])
        e_idx = torch.tensor([entity_dict["e_idx"]], dtype=torch.long)
        input_dict["entity_inputs"] = TensorDict({}, batch_size=[1])
        if self.tokenizer is not None:
            e_inputs = self.tokenizer(text=entity_dict["description"],
                        add_special_tokens=True,
                        max_length=self.args.e_max_length,################
                        padding='max_length',
                        return_token_type_ids=True,
                        truncation=True,
                        return_tensors="pt")
            for key, value in e_inputs.items():
                input_dict["entity_inputs"][key] = value
        input_dict["entity_inputs"]["e_idx"] = e_idx
        # input_dict["entity_inputs"]["sample_neighborhood"] = self.get_neighborhood(e_idx, self.args.neighborhood_sample_K, padding=True).unsqueeze(0)
        if self.args.e_neighborhood:
            input_dict["entity_inputs"]["sample_neighborhood"], input_dict["entity_inputs"]["sample_neighborhood_mask"] = self.get_neighborhood(e_idx, self.args.neighborhood_sample_K)

        prompt_tokens = [ '[soft_A1]', '[soft_A2]', '[soft_A3]', '[soft_A4]', '[soft_A5]','[CLS_A]',
                           '[soft_B1]', '[soft_B2]', '[soft_B3]', '[soft_B4]', '[soft_B5]','[CLS_B]']

        hw_text = ' '.join( prompt_tokens + [entity_dict['ori_desc']] )
        hw_encoding = self.hw_tokenizer(
            hw_text,
            add_special_tokens=True,
            max_length=self.args.e_max_length,################
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt"
            )
        for key, value in hw_encoding.items():
            key=f"hw_{key}"
            input_dict["entity_inputs"][key] = value
            

        return input_dict
    
    def get_neighborhood(self, e_idx, sample_k):
        neighborhood = self.entity_neighborhood[e_idx].T
        mask = torch.ones((1, sample_k), dtype=torch.long)
        if sample_k < neighborhood.size(0):
            sample_neighborhood_index = torch.randint(low=0, high=neighborhood.size(0), size=(sample_k,))
            neighborhood = neighborhood[sample_neighborhood_index]
        else:
            pad = torch.tensor((sample_k-neighborhood.size(0))*[[len(self), self.entity_neighborhood.relation_size]], dtype=torch.long)
            neighborhood = torch.concat((neighborhood, pad), dim=0)
            mask[:, -pad.size(0):] = 0
        
        return neighborhood.unsqueeze(0), mask

class Relations(torch.utils.data.dataset.Dataset):

    def __init__(self, args, data_dir="./data/FB15k-237/"):
        self.args = args
        # self.relations_data = pd.read_table(data_dir+'relations_data.txt', header=None, dtype=str).values.tolist()#['Relation','Name'], dtype=np.ndarray
        self.relations_data = get_table(data_dir+'relations_data.txt')
        self.relations_data = TorchShmSerializedList(self.relations_data)
        self.relations2index = {r[0]:i for i,r in enumerate(self.relations_data)}
        with open('./ana_1/relation_dict.txt', 'w', encoding='utf-8') as f:
            json.dump(
                self.relations2index,
                f,
                indent=1,
                ensure_ascii=False,
            )
        
    def get_idx(self, relation):
        return self.relations2index[relation]

    def get_name(self, index):
        if index>=len(self.relations_data):
            name = f"inverse {self.relations_data[index%len(self.relations_data)][1]}"
        else:
            name = self.relations_data[index][1]
        return name

    def __len__(self):
        return len(self.relations_data)

    def __getitem__(self, index):
        relation_dict = {
            "id":index,
            "name":self.get_name(index),
        }
        return relation_dict

class Konwledge_Graph(torch.utils.data.dataset.Dataset):

    def __init__(self, args, data_dir="./data/FB15k-237/", data_type = "train", reverse=False, entities=None, relations=None, KG=None, add_neighbor_name=False, add_hr_vocab=False, tokenizer=None, hw_tokenizer=None):
        self.args = args
        self.data_type = data_type
        self.hw_tokenizer = hw_tokenizer
        if data_type=="train" and KG is None:
            print("加载train_data")
            self.entities = entities
            self.relations = relations
            self.tokenizer = tokenizer
            

            if self.args.save_intermediate_files and os.path.isfile(self.args.data_dir+'triples.pkl'):
                with open(self.args.data_dir+'triples.pkl', 'rb') as file:
                    self.triples = pickle.load(file)
            else:
                self.triples = self.load_data(data_dir, data_type, reverse=False)# self.triples [(h_idx, r_idx, t_idx)]
                self.triples.extend(self.load_data(data_dir, data_type, reverse=True))
                self.triples = TorchShmSerializedList(self.triples, need_pickle=False)
                if self.args.save_intermediate_files :
                    with open(self.args.data_dir+'triples.pkl', 'wb') as file:
                        pickle.dump(self.triples, file)
            assert type(self.triples) == TorchShmSerializedList
            self.triples.moved_to_shared_memory()
            print("triples 加载完成")

            if self.args.save_intermediate_files and os.path.isfile(self.args.data_dir+'train_serialized_hr_vocab.pkl'):
                with open(self.args.data_dir+'train_serialized_hr_vocab.pkl', 'rb') as file:
                    self.train_serialized_hr_vocab = pickle.load(file)
            else:
                self.train_serialized_hr_vocab = [defaultdict(set) for _ in range(len(self.entities))]
                for i in tqdm.tqdm(range(len(self.triples))):
                    self.train_serialized_hr_vocab[self.triples[i][0].item()][self.triples[i][1].item()].add(self.triples[i][2].item())
                self.train_serialized_hr_vocab = SerializedHeadRelationVocab(self.train_serialized_hr_vocab, relation_size=len(self.relations)*2)
                if self.args.save_intermediate_files :
                    with open(self.args.data_dir+'train_serialized_hr_vocab.pkl', 'wb') as file:
                        pickle.dump(self.train_serialized_hr_vocab, file)
            assert type(self.train_serialized_hr_vocab) == SerializedHeadRelationVocab
            self.train_serialized_hr_vocab.moved_to_shared_memory()
            print("train_serialized_hr_vocab 加载完成")



            if self.args.save_intermediate_files and os.path.isfile(self.args.data_dir+'entity_neighborhood.pkl'):
                with open(self.args.data_dir+'entity_neighborhood.pkl', 'rb') as file:
                    self.entity_neighborhood = pickle.load(file)
            else:
                self.entity_neighborhood = [[[] for _ in range(2)] for _ in range(len(self.entities))]
                for i in tqdm.tqdm(range(len(self.triples))):
                    if len(self.entity_neighborhood[self.triples[i][2].item()]):
                        self.entity_neighborhood[self.triples[i][2].item()][0].append(self.triples[i][0].item()) # h_
                        self.entity_neighborhood[self.triples[i][2].item()][1].append(self.triples[i][1].item()) # r_
                self.entity_neighborhood = SerializedEntitiyNeighborhood(self.entity_neighborhood, relation_size=len(self.relations)*2)
                if self.args.save_intermediate_files :
                    with open(self.args.data_dir+'entity_neighborhood.pkl', 'wb') as file:
                        pickle.dump(self.entity_neighborhood, file)
            assert type(self.entity_neighborhood) == SerializedEntitiyNeighborhood
            self.entity_neighborhood.moved_to_shared_memory()
            print("entity_neighborhood 加载完成")

            if add_neighbor_name:
                self.entities.set_neighbors(self.entity_neighborhood)
            self.add_neighbor_name = add_neighbor_name
            # self.test_serialized_hr_vocab = [defaultdict(set) for _ in range(len(self.entities))]
            self.test_serialized_hr_vocab = defaultdict(set)
        else:
            self.entities = KG.entities
            self.relations = KG.relations
            self.tokenizer = KG.tokenizer
            self.tail_tokenizer = KG.entities.tokenizer
            self.triples = self.load_data(data_dir, data_type, reverse=reverse)# self.triples [(h_idx, r_idx, t_idx)]
            

            # self.all_hr_vocab = KG.all_hr_vocab
            # self.train_hr_vocab = KG.train_hr_vocab
            # self.neighbors = KG.neighbors

            self.test_serialized_hr_vocab = KG.test_serialized_hr_vocab
            if add_hr_vocab:
                for triple in self.triples:
                    # self.all_hr_vocab[(triple[0], triple[1])].append(triple[2])
                    self.test_serialized_hr_vocab[(triple[0], triple[1])].add(triple[2])

            self.add_neighbor_name = KG.add_neighbor_name
            self.train_serialized_hr_vocab = KG.train_serialized_hr_vocab
            self.entity_neighborhood = KG.entity_neighborhood
            self.triples = TorchShmSerializedList(self.triples, need_pickle=False)

        
    def load_data(self, data_dir, data_type="train", reverse=False) -> list:
        # data = pd.read_table(f'{data_dir}{data_type}.tsv', header=None, dtype=str).values.tolist()
        data = get_table(f'{data_dir}{data_type}.tsv')
        
        if not reverse:
            for triple in data:
                triple[0] = self.entities.get_idx(triple[0])
                triple[1] = self.relations.get_idx(triple[1])
                triple[2] = self.entities.get_idx(triple[2])
        else:
            for triple in data:
                tmp = triple[0]
                triple[0] = self.entities.get_idx(triple[2])
                triple[1] = self.relations.get_idx(triple[1]) + len(self.relations)
                triple[2] = self.entities.get_idx(tmp)
        return data

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        h_idx = self.triples[index][0].item()
        r_idx = self.triples[index][1].item()
        t_idx = self.triples[index][2].item()
        h_name = self.entities.get_name(h_idx)
        h_description = self.entities.get_description(h_idx)
        r_name = self.relations.get_name(r_idx)
        t_name = self.entities.get_name(t_idx)
        t_description = self.entities.get_description(t_idx)
                        
        ####可能很慢
        if self.add_neighbor_name and len(h_description.split())<20:
            h_neighbors_name = [self.entities.get_name(neighbor) for neighbor in set(self.entity_neighborhood[h_idx][0][:30].tolist()) if neighbor!=t_idx]# 不将尾实体作为头实体neighbor输入
            h_description = h_description + "; neighbor: " + ",".join(h_neighbors_name)
            del h_neighbors_name
        if self.add_neighbor_name and len(t_description.split())<20:
            t_neighbors_name = [self.entities.get_name(neighbor) for neighbor in set(self.entity_neighborhood[t_idx][0][:30].tolist()) if neighbor!=h_idx]# 不将头实体作为尾实体neighbor输入
            t_description = t_description + "; neighbor: " + ",".join(t_neighbors_name)
            del t_neighbors_name
       
        triple_dict = {
            "h_idx":h_idx,
            "r_idx":r_idx,
            "t_idx":t_idx,
            "h_name":h_name,
            "h_description":h_description,
            "h_ori_desc": f"{h_name}，{self.entities.get_description(h_idx)}",
            "r_name":r_name,
            "t_name":t_name,
            "t_description":t_description,
            "t_ori_desc": f"{t_name}，{self.entities.get_description(t_idx)}",
        }
        input_dict = self.vectorize(triple_dict)
        del triple_dict
        return input_dict
    
    def vectorize(self, triple_dict):
        input_dict = {}
        h_idx = torch.tensor([triple_dict["h_idx"]], dtype=torch.long)
        r_idx = torch.tensor([triple_dict["r_idx"]], dtype=torch.long)
        t_idx = torch.tensor([triple_dict["t_idx"]], dtype=torch.long)
        if self.args.task=="train":
            input_dict['targets'] = self.train_serialized_hr_vocab[h_idx, r_idx]
        input_dict["hr_inputs"] = {}
        input_dict["t_inputs"] = {}
        input_dict["pt_inputs"] = {}
        if self.tokenizer is not None:
            hr_inputs = self.tokenizer(text=triple_dict["h_description"],
                                text_pair=triple_dict["r_name"],
                                add_special_tokens=True,
                                max_length=self.args.hr_max_length,###############
                                padding='max_length',
                                return_token_type_ids=True,
                                truncation=True,
                                return_tensors="pt")
            for key, value in hr_inputs.items():
                input_dict["hr_inputs"][key] = value
            del hr_inputs
        input_dict["hr_inputs"]["h_idx"] = h_idx
        input_dict["hr_inputs"]["r_idx"] = r_idx
        input_dict["hr_inputs"]["mask_idx"] = t_idx
        # input_dict["hr_inputs"]["sample_neighborhood"] = self.get_neighborhood(h_idx, self.args.neighborhood_sample_K, padding=True).unsqueeze(0)
        if self.args.hr_neighborhood:
            input_dict["hr_inputs"]["sample_neighborhood"], input_dict["hr_inputs"]["sample_neighborhood_mask"] = self.get_neighborhood(h_idx, t_idx, (r_idx+len(self.relations))%(2*len(self.relations)), self.args.neighborhood_sample_K)

        prompt_tokens = [ '[soft_A1]', '[soft_A2]', '[soft_A3]', '[soft_A4]', '[soft_A5]','[CLS_A]',
                           '[soft_B1]', '[soft_B2]', '[soft_B3]', '[soft_B4]', '[soft_B5]','[CLS_B]']

        h_hw_text = ' '.join( prompt_tokens + [triple_dict['h_ori_desc']] )
        h_hw_encoding = self.hw_tokenizer(
            h_hw_text,
            add_special_tokens=True,
            max_length=self.args.e_max_length,################
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt"
            )
        for key, value in h_hw_encoding.items():
            key=f"hw_{key}"
            input_dict["hr_inputs"][key] = value
            
            


        if self.tokenizer is not None:
            t_inputs = self.tokenizer(text=triple_dict["t_description"],
                                add_special_tokens=True,
                                max_length=self.args.e_max_length,################
                                padding='max_length',
                                return_token_type_ids=True,
                                truncation=True,
                                return_tensors="pt")
            for key, value in t_inputs.items():
                input_dict["t_inputs"][key] = value
                input_dict["pt_inputs"][key] = value
            del t_inputs
        input_dict["t_inputs"]["e_idx"] = t_idx
        input_dict["t_inputs"]["mask_idx"] = h_idx
        # input_dict["t_inputs"]["sample_neighborhood"] = self.get_neighborhood(t_idx, self.args.neighborhood_sample_K, padding=True).unsqueeze(0)
        if self.args.e_neighborhood:
            input_dict["t_inputs"]["sample_neighborhood"], input_dict["t_inputs"]["sample_neighborhood_mask"] = self.get_neighborhood(t_idx, h_idx, r_idx, self.args.neighborhood_sample_K)

        prompt_tokens = [ '[soft_A1]', '[soft_A2]', '[soft_A3]', '[soft_A4]', '[soft_A5]','[CLS_A]',
                           '[soft_B1]', '[soft_B2]', '[soft_B3]', '[soft_B4]', '[soft_B5]','[CLS_B]']

        t_hw_text = ' '.join(  prompt_tokens + [triple_dict['t_ori_desc']])
        t_hw_encoding = self.hw_tokenizer(
            t_hw_text,
            add_special_tokens=True,
            max_length=self.args.e_max_length,################
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt"
            )
        for key, value in t_hw_encoding.items():
            key=f"hw_{key}"
            input_dict["t_inputs"][key] = value
            


        # input_dict["pt_inputs"]["e_idx"] = t_idx
        # input_dict["pt_inputs"]["mask_idx"] = h_idx
        # if self.args.e_neighborhood:
        #     input_dict["pt_inputs"]["sample_neighborhood"], input_dict["pt_inputs"]["sample_neighborhood_mask"] = self.entities.get_neighborhood(t_idx, self.args.neighborhood_sample_K)
                
        return input_dict
    
    def get_neighborhood(self, e_idx, mask_h_idx, mask_r_idx, sample_k):
        neighborhood = self.entity_neighborhood[e_idx].T
        mask = torch.ones((1, sample_k), dtype=torch.long)
        if sample_k < neighborhood.size(0):
            sample_neighborhood_index = torch.randint(low=0, high=neighborhood.size(0), size=(sample_k,))
            neighborhood = neighborhood[sample_neighborhood_index]
        else:
            pad = torch.tensor((sample_k-neighborhood.size(0))*[[len(self.entities), self.entity_neighborhood.relation_size]], dtype=torch.long)
            neighborhood = torch.concat((neighborhood, pad), dim=0)
            mask[:, -pad.size(0):] = 0
        if self.args.task == "train":
            hr_idx = torch.where(torch.logical_and(neighborhood[:,0]==mask_h_idx, neighborhood[:,1]==mask_r_idx))[0]
            if len(hr_idx)>0:
                mask[:, hr_idx] = 0
                neighborhood[hr_idx, 0] = len(self.entities)
                neighborhood[hr_idx, 1] = self.entity_neighborhood.relation_size
            
            # add_mask = torch.randint(low=0, high=sample_k, size=(sample_k//10,))
            add_mask = torch.rand(size=(sample_k,))<0.1
            mask[:, add_mask] = 0
            neighborhood[add_mask, 0] = len(self.entities)
            neighborhood[add_mask, 1] = self.entity_neighborhood.relation_size
        return neighborhood.unsqueeze(0), mask

class negative_sampling_Entities(torch.utils.data.dataset.Dataset):

    def __init__(self, _len, entities:Entities):
        self.entities = entities
        self._len = _len
    
    def __len__(self):
        return self._len

    def __getitem__(self, index):
        index = index%len(self.entities)
        input_dict = self.entities[index]
        if self.entities.args.e_neighborhood:
            neighborhood = input_dict["entity_inputs"]["sample_neighborhood"]
            mask = input_dict["entity_inputs"]["sample_neighborhood_mask"]
            # add_mask = torch.randint(low=0, high=self.entities.args.neighborhood_sample_K, size=(self.entities.args.neighborhood_sample_K//10,))
            add_mask = torch.rand(size=(self.entities.args.neighborhood_sample_K,))<0.1
            mask[:, add_mask] = 0
            neighborhood[0, add_mask, 0] = len(self.entities)
            neighborhood[0, add_mask, 1] = self.entities.entity_neighborhood.relation_size
        return input_dict


def collate(batch_data: List[TensorDict]) -> TensorDict:
    def _collate(batch_data):
        if len(batch_data)==1:
            return batch_data[0]
        batch_data_ = {k:[v]for k,v in batch_data[0].items()}
        for data in batch_data[1:]:
            for k,v in data.items():
                batch_data_[k].append(v)
        for k,v in batch_data_.items():
            if k=="targets":
                continue
            if type(v[0]) is torch.Tensor:
                batch_data_[k] = torch.concat(v, dim=0)
            elif type(v[0]) is list:
                # batch_data_[k] = [vi[0] for vi in v]
                batch_data_[k] = v
            else:
                batch_data_[k] = _collate(v)
        return batch_data_

    batch_data = _collate(batch_data)
    # batch_data = torch.cat(batch_data, dim=0)
    return batch_data

def data_to_device(data, device):
    if type(data) is torch.Tensor or type(data) is BatchEncoding or type(data) is TensorDict:
        data = data.to(device)
    elif type(data) is dict:
        for key, value in data.items():
            data[key] = data_to_device(value, device)
    elif type(data) is list:
        for i, item in enumerate(data):
            data[i] = data_to_device(item, device)
    return data

def batch_split(batch_data, batch_size, actual_batch_size):
    def split_list(input_list, group_size):
        divided_list = []
        for i in range(0, len(input_list), group_size):
            group = input_list[i:i + group_size]
            divided_list.append(group)
        return divided_list

    if type(batch_data) is torch.Tensor or type(batch_data) is TensorDict:
        new_data = batch_data.split(actual_batch_size, dim=0)
    elif type(batch_data) is list:
        new_data = split_list(batch_data, actual_batch_size)
    elif type(batch_data) is dict:
        new_data = [{} for i in range(math.ceil(batch_size/actual_batch_size))]
        for key, value in batch_data.items():
            data_list = batch_split(value, batch_size, actual_batch_size)
            for i, data in enumerate(data_list):
                new_data[i][key] = data
    else:
        raise ValueError("")
    return new_data

def get_sim(x_t, y_t:torch.Tensor, norm = False, split_batch=2**18):
    if norm:
        x_t = F.normalize(x_t, dim=-1)
    if y_t.size(0)<=split_batch:
        if norm:
            y_t = F.normalize(y_t, dim=-1)
        else:
            score = x_t @ y_t.T
    else:
        # x_t = x_t.to("cuda:1")
        score = []
        for i in range(0, y_t.size(0), split_batch):
            _y_t = y_t[i:i+split_batch]
            # _y_t = _y_t.to("cuda:1")
            if norm:
                score.append((x_t @ F.normalize(_y_t, dim=-1).transpose(0,1)))#.to("cpu"))
            else:
                score.append((x_t @ _y_t.T))#.to("cpu"))
        score = torch.concat(score, dim=1)
    return score


import logging
import os
def get_logger(log_file):
    if log_file is None:
        return None

    if os.path.exists(log_file):
        os.remove(log_file)

    test_log = logging.FileHandler(log_file,'a',encoding='utf-8')
    test_log.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(process)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(process)s - %(message)s')
    test_log.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(test_log)

    return logger

import random
import numpy as np
def set_seed(seed):
    torch.backends.cudnn.deterministic = True 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 

def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
