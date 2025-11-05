import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, average_precision_score
import numpy as np
from tqdm import tqdm
import os
import json
from collections import defaultdict
from sklearn.utils import resample



# 定义模型
class MultiTaskTextClassifier(nn.Module):
    def __init__(self, model_name, n_multi_labels, n_single_labels, prompt_tokens, tokenizer, dropout_p=0.3):
        """
        多任务文本分类器，包含多标签和单标签分类。

        Args:
            model_name (str): 预训练模型名称或路径。
            n_multi_labels (int): 多标签数量。
            n_single_labels (int): 单标签数量。
            prompt_tokens (list of str): 软提示的特殊标记列表。
            tokenizer (BertTokenizer): BERT tokenizer。
            dropout_p (float): Dropout 概率。
        """
        super(MultiTaskTextClassifier, self).__init__()
        self.n_multi_labels = n_multi_labels
        self.n_single_labels = n_single_labels
        self.soft_prompt_length = (len(prompt_tokens) - 2) // 2  # 除去 [CLS_A], [CLS_B]，每组5个

        self.bert = BertModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(len(tokenizer))  # 扩展词汇表以包含新添加的软提示 token

        # 初始化软提示对应的嵌入
        # 获取初始化嵌入的 token ID
        cls_token_id = tokenizer.cls_token_id
        soft_prompt_initial_mappings_A = {
            '[soft_A1]': 2364,  
            '[soft_A2]': 7367,
            '[soft_A3]':  4168,
            '[soft_A4]':  4168,
            '[soft_A5]': 1024
        }
        soft_prompt_initial_mappings_B = {
            '[soft_B1]': 2035,
            '[soft_B2]': 7367,
            '[soft_B3]': 4168,
            '[soft_B4]': 7834,
            '[soft_B5]': 1024
        }

        with torch.no_grad():
            cls_a_id = tokenizer.convert_tokens_to_ids('[CLS_A]')
            cls_b_id = tokenizer.convert_tokens_to_ids('[CLS_B]')
            bert_cls_embedding = self.bert.embeddings.word_embeddings.weight[cls_token_id]
            
            self.bert.embeddings.word_embeddings.weight[cls_a_id] = bert_cls_embedding.clone().detach()
            self.bert.embeddings.word_embeddings.weight[cls_b_id] = bert_cls_embedding.clone().detach()
            print(cls_a_id,cls_b_id,cls_token_id,bert_cls_embedding.size())

            # 初始化 [soft_A1]-[soft_A5] 为 “主”、“义”、“原”、“为”、“：”
            for token, word in soft_prompt_initial_mappings_A.items():
                token_id = tokenizer.convert_tokens_to_ids(token)
                word_id = word
                print(token,word,word_id,token_id,self.bert.embeddings.word_embeddings.weight[token_id].size())
                if word_id == tokenizer.unk_token_id:
                    # 如果词汇表中没有该词，则随机初始化
                    print(f"Warning: Word '{word}' not found in tokenizer vocabulary. Initializing '[{token}]' randomly.")
                    embedding = torch.randn_like(self.bert.embeddings.word_embeddings.weight[token_id])
                else:
                    # 使用 "主"、"义" 等词的 embedding 初始化软提示
                    embedding = self.bert.embeddings.word_embeddings.weight[word_id]
                
                self.bert.embeddings.word_embeddings.weight[token_id] = embedding.clone().detach()
                print(token,word,word_id,token_id,self.bert.embeddings.word_embeddings.weight[token_id].size())

            # 初始化 [soft_B1]-[soft_B5] 为 “义”、“原”、“信”、“息”、“：”
            for token, word in soft_prompt_initial_mappings_B.items():
                token_id = tokenizer.convert_tokens_to_ids(token)
                word_id = word
                print(token,word,word_id,token_id,self.bert.embeddings.word_embeddings.weight[token_id].size())
                if word_id == tokenizer.unk_token_id:
                    # 如果词汇表中没有该词，则随机初始化
                    print(f"Warning: Word '{word}' not found in tokenizer vocabulary. Initializing '[{token}]' randomly.")
                    embedding = torch.randn_like(self.bert.embeddings.word_embeddings.weight[token_id])
                else:
                    # 使用 "义"、"原" 等词的 embedding 初始化软提示
                    embedding = self.bert.embeddings.word_embeddings.weight[word_id]
                self.bert.embeddings.word_embeddings.weight[token_id] = embedding.clone().detach()
                print(token,word,word_id,token_id,self.bert.embeddings.word_embeddings.weight[token_id].size())

        self.dropout = nn.Dropout(dropout_p)

        # 多标签分类器
        self.multi_label_classifier = nn.Linear(self.bert.config.hidden_size, n_multi_labels)

        # 单标签分类器
        self.single_label_classifier = nn.Linear(self.bert.config.hidden_size, n_single_labels)

        # 定义损失函数
        self.multi_label_loss_fn = nn.BCEWithLogitsLoss()
        self.single_label_loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, multi_labels=None, single_labels=None):
        """
        前向传播。

        Args:
            input_ids (Tensor): 输入的 token IDs。
            attention_mask (Tensor): 注意力掩码。
            multi_labels (Tensor, optional): 多标签的真实值。
            single_labels (Tensor, optional): 单标签的真实值。

        Returns:
            Tuple: (multi_loss, single_loss, multi_logits, single_logits)
        """
        # 获取 BERT 输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        # 确定 [CLS_A] 和 [CLS_B] 的位置
        # 假设输入格式: [CLS] [soft_A1]...[soft_A5][CLS_A] [soft_B1]...[soft_B5] [CLS_B]文本 [SEP]
        cls_a_position = self.soft_prompt_length+1
        cls_b_position = 1 + 1 + self.soft_prompt_length*2  # [CLS_A] + 5 soft tokens

        # 提取 [CLS_A] 和 [CLS_B] 的隐藏状态
        cls_a_hidden = last_hidden_state[:, cls_a_position, :]  # (batch_size, hidden_size)
        cls_b_hidden = last_hidden_state[:, cls_b_position, :]  # (batch_size, hidden_size)
        #print(cls_a_hidden.size())

        # 分类头
        single_logits = self.single_label_classifier(self.dropout(cls_a_hidden))
        multi_logits = self.multi_label_classifier(self.dropout(cls_b_hidden))
        

        # 计算损失
        multi_loss = 0
        single_loss = 0
        if multi_labels is not None:
            multi_loss = self.multi_label_loss_fn(multi_logits, multi_labels)
        if single_labels is not None:
            single_loss = self.single_label_loss_fn(single_logits, single_labels)

        return multi_loss, single_loss, multi_logits, single_logits
    def predict(self, input_ids, attention_mask, multi_labels=None, single_labels=None):
        """
        前向传播。

        Args:
            input_ids (Tensor): 输入的 token IDs。
            attention_mask (Tensor): 注意力掩码。
            multi_labels (Tensor, optional): 多标签的真实值。
            single_labels (Tensor, optional): 单标签的真实值。

        Returns:
            Tuple: (multi_loss, single_loss, multi_logits, single_logits)
        """
        # 获取 BERT 输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        # 确定 [CLS_A] 和 [CLS_B] 的位置
        # 假设输入格式: [CLS] [soft_A1]...[soft_A5][CLS_A] [soft_B1]...[soft_B5] [CLS_B]文本 [SEP]
        cls_a_position = self.soft_prompt_length+1
        cls_b_position = 1 + 1 + self.soft_prompt_length*2  # [CLS_A] + 5 soft tokens

        # 提取 [CLS_A] 和 [CLS_B] 的隐藏状态
        cls_a_hidden = last_hidden_state[:, cls_a_position, :]  # (batch_size, hidden_size)
        cls_b_hidden = last_hidden_state[:, cls_b_position, :]  # (batch_size, hidden_size)
        #print(cls_a_hidden.size())

        # 分类头
        single_logits = self.single_label_classifier(self.dropout(cls_a_hidden))
        multi_logits = self.multi_label_classifier(self.dropout(cls_b_hidden))
        
        return cls_a_hidden, cls_b_hidden, multi_logits, single_logits


     



def load_sp_model_and_tokenizer(args):
    # 加载和准备数据
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # 定义软提示的特殊标记
    # [CLS_A] 和 [soft_A1]-[soft_A5] 分别用于多标签任务
    # [CLS_B] 和 [soft_B1]-[soft_B5] 分别用于单标签任务
    soft_prompt_tokens = [ '[soft_A1]', '[soft_A2]', '[soft_A3]', '[soft_A4]', '[soft_A5]','[CLS_A]',
                           '[soft_B1]', '[soft_B2]', '[soft_B3]', '[soft_B4]', '[soft_B5]','[CLS_B]']
    print("Soft prompt tokens:", soft_prompt_tokens)

    # 添加软提示特殊标记到 tokenizer
    tokenizer.add_tokens(soft_prompt_tokens, special_tokens=True)
    

    n_multi_labels = 2042
    n_single_labels = 2042
    print(f"Number of multi-labels: {n_multi_labels}")
    print(f"Number of single-labels: {n_single_labels}")

    # 初始化模型
    model = MultiTaskTextClassifier(
        model_name=args.model_name,
        n_multi_labels=n_multi_labels,
        n_single_labels=n_single_labels,
        prompt_tokens=soft_prompt_tokens,
        tokenizer=tokenizer,  # 传递 tokenizer 以便模型内初始化
        dropout_p=args.dropout
    )
    model.load_state_dict(torch.load('./hw_model/best_model_multi.bin'))
    model = model.to(args.device)
    

    # 获取 soft prompt token IDs
    soft_prompt_ids = tokenizer.convert_tokens_to_ids(soft_prompt_tokens)  # [soft_A1]-[soft_A5] 和 [soft_B1]-[soft_B5]
    print('soft_prompt_ids',soft_prompt_ids)
    return tokenizer,model

class Hw_Model_Args:
    def __init__(self):
        self.model_name='./sp_pretrain_model/'
        self.max_length=40
        self.dropout=0.5
        self.device ='cuda:0'
        self.soft_prompt_length=5
   

def load_hw_main():
    hw_model_args = Hw_Model_Args()

    return load_sp_model_and_tokenizer(hw_model_args)
