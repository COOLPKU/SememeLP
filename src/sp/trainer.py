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


class TextDataset(Dataset):
    def __init__(self, texts, label_lists, single_labels, tokenizer, max_length, prompt_tokens):
        """


        Args:
            texts (list of str): List of text data.
            label_lists (list of list of int): Multi-label list, where each inner list represents the multi-labels of one sample.
            single_labels (list of int): Single-label list, where each element represents the single label of one sample.
            tokenizer (BertTokenizer): BERT tokenizer.
            max_length (int): Maximum text length.
            prompt_tokens (list of str): List of special tokens for soft prompts.
        """
        self.texts = texts
        self.label_lists = label_lists
        self.single_labels = single_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_tokens = prompt_tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_list = self.label_lists[idx]
        single_label = self.single_labels[idx]

        # 构建包含软提示的完整文本
        # 输入格式: [CLS] [CLS_A] [soft_A1]...[soft_A5] [CLS_B] [soft_B1]...[soft_B5] 文本 [SEP]
        full_text = ' '.join(['[CLS]'] + self.prompt_tokens + [text] + ['[SEP]'])

        encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,  # 已经手动添加
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'multi_labels': torch.tensor(label_list, dtype=torch.float),
            'single_label': torch.tensor(single_label, dtype=torch.long)
        }


class MultiTaskTextClassifier(nn.Module):
    def __init__(self, model_name, n_multi_labels, n_single_labels, prompt_tokens, tokenizer, dropout_p=0.3):
        """


        Args:
            model_name (str): Name or path of the pre-trained model.
            n_multi_labels (int): Number of multi-labels.
            n_single_labels (int): Number of single labels.
            prompt_tokens (list of str): List of special tokens for soft prompts.
            tokenizer (BertTokenizer): BERT tokenizer.
            dropout_p (float): Dropout probability.
        """
        super(MultiTaskTextClassifier, self).__init__()
        self.n_multi_labels = n_multi_labels
        self.n_single_labels = n_single_labels
        self.soft_prompt_length = (len(prompt_tokens) - 2) // 2

        self.bert = BertModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(len(tokenizer))


        cls_token_id = tokenizer.cls_token_id
        soft_prompt_initial_mappings_A = {
            '[soft_A1]': '主',
            '[soft_A2]': '义',
            '[soft_A3]': '原',
            '[soft_A4]': '为',
            '[soft_A5]': ':'
        }
        soft_prompt_initial_mappings_B = {
            '[soft_B1]': '义',
            '[soft_B2]': '原',
            '[soft_B3]': '信',
            '[soft_B4]': '息',
            '[soft_B5]': ':'
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
                word_id = tokenizer.convert_tokens_to_ids(word)
                print(token,word,word_id,token_id,self.bert.embeddings.word_embeddings.weight[token_id].size())
                if word_id == tokenizer.unk_token_id:

                    print(f"Warning: Word '{word}' not found in tokenizer vocabulary. Initializing '[{token}]' randomly.")
                    embedding = torch.randn_like(self.bert.embeddings.word_embeddings.weight[token_id])
                else:

                    embedding = self.bert.embeddings.word_embeddings.weight[word_id]
                
                self.bert.embeddings.word_embeddings.weight[token_id] = embedding.clone().detach()
                print(token,word,word_id,token_id,self.bert.embeddings.word_embeddings.weight[token_id].size())

            # 初始化 [soft_B1]-[soft_B5] 为 “义”、“原”、“信”、“息”、“：”
            for token, word in soft_prompt_initial_mappings_B.items():
                token_id = tokenizer.convert_tokens_to_ids(token)
                word_id = tokenizer.convert_tokens_to_ids(word)
                print(token,word,word_id,token_id,self.bert.embeddings.word_embeddings.weight[token_id].size())
                if word_id == tokenizer.unk_token_id:
                    # 如果词汇表中没有该词，则随机初始化
                    print(f"Warning: Word '{word}' not found in tokenizer vocabulary. Initializing '[{token}]' randomly.")
                    embedding = torch.randn_like(self.bert.embeddings.word_embeddings.weight[token_id])
                else:

                    embedding = self.bert.embeddings.word_embeddings.weight[word_id]
                self.bert.embeddings.word_embeddings.weight[token_id] = embedding.clone().detach()
                print(token,word,word_id,token_id,self.bert.embeddings.word_embeddings.weight[token_id].size())

        self.dropout = nn.Dropout(dropout_p)


        self.multi_label_classifier = nn.Linear(self.bert.config.hidden_size, n_multi_labels)

        self.single_label_classifier = nn.Linear(self.bert.config.hidden_size, n_single_labels)


        self.multi_label_loss_fn = nn.BCEWithLogitsLoss()
        self.single_label_loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, multi_labels=None, single_labels=None):
        """
        前向传播。

        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention mask.
            multi_labels (Tensor, optional): Ground truth of multi-labels.
            single_labels (Tensor, optional): Ground truth of single labels.

        Returns:
            Tuple: (multi_loss, single_loss, multi_logits, single_logits)
        """

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)


        # [CLS] [soft_A1]...[soft_A5][CLS_A] [soft_B1]...[soft_B5] [CLS_B] text [SEP]
        cls_a_position = self.soft_prompt_length+1
        cls_b_position = 1 + 1 + self.soft_prompt_length*2  # [CLS_A] + 5 soft tokens


        cls_a_hidden = last_hidden_state[:, cls_a_position, :]  # (batch_size, hidden_size)
        cls_b_hidden = last_hidden_state[:, cls_b_position, :]  # (batch_size, hidden_size)
        #print(cls_a_hidden.size())


        single_logits = self.single_label_classifier(self.dropout(cls_a_hidden))
        multi_logits = self.multi_label_classifier(self.dropout(cls_b_hidden))
        

        multi_loss = 0
        single_loss = 0
        if multi_labels is not None:
            multi_loss = self.multi_label_loss_fn(multi_logits, multi_labels)
        if single_labels is not None:
            single_loss = self.single_label_loss_fn(single_logits, single_labels)

        return multi_loss, single_loss, multi_logits, single_logits

# 过采样数据
def oversample_data(texts, label_lists, single_labels, min_count=10):
    """
    For each label, ensure there are at least min_count positive samples by duplicating existing positive samples (oversampling).
    Args:
        texts (list of str): List of text
        data.label_lists (list of list of int): List of multi-label
        data.single_labels (list of int): List of single-label
        data.min_count (int): Minimum number of positive samples required for each label.
    Returns:Tuple: (Augmented texts, label_lists, single_labels)
    """
    label_to_indices_multi = defaultdict(list)
    label_to_indices_single = defaultdict(list)
    for idx, (multi_label, single_label) in enumerate(zip(label_lists, single_labels)):
        for i, val in enumerate(multi_label):
            if val == 1:
                label_to_indices_multi[i].append(idx)
        label_to_indices_single[single_label].append(idx)
    
    new_texts = list(texts)
    new_label_lists = list(label_lists)
    new_single_labels = list(single_labels)
    

    for label, indices in label_to_indices_multi.items():
        if len(indices) > 0 and len(indices) < min_count:
            num_to_add = min_count - len(indices)
            sampled_indices = resample(indices, replace=True, n_samples=num_to_add)
            for idx in sampled_indices:
                new_texts.append(texts[idx])
                new_label_lists.append(label_lists[idx])
                new_single_labels.append(single_labels[idx])
    

    for label, indices in label_to_indices_single.items():
        if len(indices) > 0 and len(indices) < min_count:
            num_to_add = min_count - len(indices)
            sampled_indices = resample(indices, replace=True, n_samples=num_to_add)
            for idx in sampled_indices:
                new_texts.append(texts[idx])
                new_label_lists.append(label_lists[idx])
                new_single_labels.append(single_labels[idx])
    
    return new_texts, new_label_lists, new_single_labels

def eval_model(model, data_loader, device):
    model.eval()
    losses_multi = []
    losses_single = []
    preds_multi = []
    preds_single = []
    targets_multi = []
    targets_single = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            multi_labels = batch['multi_labels'].to(device)
            single_labels = batch['single_label'].to(device)

            multi_loss, single_loss, multi_logits, single_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                multi_labels=multi_labels,
                single_labels=single_labels
            )

            losses_multi.append(multi_loss.item())
            losses_single.append(single_loss.item())

            preds_multi.append(torch.sigmoid(multi_logits).cpu().numpy())
            preds_single.append(torch.softmax(single_logits, dim=1).cpu().numpy())

            targets_multi.append(multi_labels.cpu().numpy())
            targets_single.append(single_labels.cpu().numpy())


     
    preds_multi = np.concatenate(preds_multi, axis=0)

    targets_multi = np.concatenate(targets_multi, axis=0)

    if np.sum(targets_multi) > 0:
        avg_precision_multi = average_precision_score(targets_multi, preds_multi, average='samples')
        f1_multi = f1_score(targets_multi, np.round(preds_multi), average='micro')
        print('macro',f1_score(targets_multi, np.round(preds_multi), average='macro'))
    else:
        print('error')
        avg_precision_multi = 0
        f1_multi = 0


    preds_single = np.concatenate(preds_single, axis=0)
    targets_single = np.concatenate(targets_single, axis=0).astype(int)
    preds_single_class = np.argmax(preds_single, axis=1)
    accuracy_single = np.mean(preds_single_class == targets_single)
    if len(np.unique(targets_single)) > 1:
        f1_single = f1_score(targets_single, preds_single_class, average='micro')
    else:
        print('error')
        f1_single = 0

    print(f"Validation Multi Loss: {np.mean(losses_multi):.4f}, MAP: {avg_precision_multi:.4f}, F1 (Multi): {f1_multi:.4f}")
    print(f"Validation Single Loss: {np.mean(losses_single):.4f}, Accuracy (Single): {accuracy_single:.4f}, F1 (Single): {f1_single:.4f}")

    return np.mean(losses_multi), avg_precision_multi, f1_multi, np.mean(losses_single), accuracy_single, f1_single



def load_data(data_path):
    print(f"Loading data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"{data_path} contains {len(data)} samples.")
    if len(data) > 0:
        print(f"Sample format: {data[0]}")
    texts = []
    label_lists = []
    single_labels = []
    
    for d in tqdm(data):
        texts.append(d['key'])
        label_lists.append(d['labels'])
        # 假设 single_label 已经作为独立字段存在
        single_label = d['main_label']
        single_labels.append(single_label)
    
    return texts, label_lists, single_labels


def train_and_evaluate(model, train_loader, dev_loader, optimizer, scheduler, device, args, best_f1_multi=0.0, best_f1_single=0.0):
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        losses_multi = []
        losses_single = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for batch in progress_bar:
            global_step += 1
            input_ids = batch['input_ids'].to(device)
            
            attention_mask = batch['attention_mask'].to(device)
            multi_labels = batch['multi_labels'].to(device)
            single_labels = batch['single_label'].to(device)

            # Forward pass
            multi_loss, single_loss, _, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                multi_labels=multi_labels,
                single_labels=single_labels
            )


            total_loss = multi_loss + args.loss_weight * single_loss


            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            

            losses_multi.append(multi_loss.item())
            losses_single.append(single_loss.item())
            progress_bar.set_postfix({
                'Train Multi Loss': f"{np.mean(losses_multi):.4f}",
                'Train Single Loss': f"{np.mean(losses_single):.4f}"
            })


            if global_step % args.eval_steps == 0:
                print(f"\n----- Step {global_step}: Starting Evaluation -----")
                val_multi_loss, val_avg_precision_multi, val_f1_multi, val_single_loss, val_accuracy_single, val_f1_single = eval_model(model, dev_loader, device)
                
                print(f"Step {global_step}: Val Multi Loss = {val_multi_loss:.4f}, MAP = {val_avg_precision_multi:.4f}, F1 (Multi) = {val_f1_multi:.4f}")
                print(f"Step {global_step}: Val Single Loss = {val_single_loss:.4f}, Accuracy (Single) = {val_accuracy_single:.4f}, F1 (Single) = {val_f1_single:.4f}")


                if val_f1_multi > best_f1_multi:
                    best_f1_multi = val_f1_multi
                    torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_multi.bin'))
                    print(f"--> Best Multi-label model updated with F1: {best_f1_multi:.4f}")

                if val_f1_single > best_f1_single:
                    best_f1_single = val_f1_single
                    torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_single.bin'))
                    print(f"--> Best Single-label model updated with F1: {best_f1_single:.4f}")


        print(f"\n===== Epoch {epoch+1} 完成 =====")
        val_multi_loss, val_avg_precision_multi, val_f1_multi, val_single_loss, val_accuracy_single, val_f1_single = eval_model(model, dev_loader, device)
        print(f"Epoch {epoch + 1}: Val Multi Loss = {val_multi_loss:.4f}, MAP = {val_avg_precision_multi:.4f}, F1 (Multi) = {val_f1_multi:.4f}")
        print(f"Epoch {epoch + 1}: Val Single Loss = {val_single_loss:.4f}, Accuracy (Single) = {val_accuracy_single:.4f}, F1 (Single) = {val_f1_single:.4f}")


        if val_f1_multi > best_f1_multi:
            best_f1_multi = val_f1_multi
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_multi.bin'))
            print(f"--> Best Multi-label model updated with F1: {best_f1_multi:.4f}")


        if val_f1_single > best_f1_single:
            best_f1_single = val_f1_single
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_single.bin'))
            print(f"--> Best Single-label model updated with F1: {best_f1_single:.4f}")




def main(args):

    tokenizer = BertTokenizer.from_pretrained(args.model_name)


    soft_prompt_tokens = [ '[soft_A1]', '[soft_A2]', '[soft_A3]', '[soft_A4]', '[soft_A5]','[CLS_A]',
                           '[soft_B1]', '[soft_B2]', '[soft_B3]', '[soft_B4]', '[soft_B5]','[CLS_B]']
    print("Soft prompt tokens:", soft_prompt_tokens)


    tokenizer.add_tokens(soft_prompt_tokens, special_tokens=True)


    train_texts, train_label_lists, train_single_labels = load_data(args.train_data_dir)
    dev_texts, dev_label_lists, dev_single_labels = load_data(args.dev_data_dir)

    # Oversampling: Ensure each label has at least min_count positive samples.
    print('过采样前', len(train_texts))
    train_texts, train_label_lists, train_single_labels = oversample_data(train_texts, train_label_lists, train_single_labels, min_count=10)
    print('过采样后', len(train_texts))


    if args.loss_type == 'weighted_bce':
        pos_weight_multi = compute_pos_weight(train_label_lists)
    else:
        pos_weight_multi = None


    train_dataset = TextDataset(train_texts, train_label_lists, train_single_labels, tokenizer, args.max_length, soft_prompt_tokens)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(train_dataset[0])

    dev_dataset = TextDataset(dev_texts, dev_label_lists, dev_single_labels, tokenizer, args.max_length, soft_prompt_tokens)
    dev_data_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    print(dev_dataset[0])

    n_multi_labels = len(train_label_lists[0])
    n_single_labels = n_multi_labels  # 假设单标签是整数编码
    print(f"Number of multi-labels: {n_multi_labels}")
    print(f"Number of single-labels: {n_single_labels}")


    model = MultiTaskTextClassifier(
        model_name=args.model_name,
        n_multi_labels=n_multi_labels,
        n_single_labels=n_single_labels,
        prompt_tokens=soft_prompt_tokens,
        tokenizer=tokenizer,  # 传递 tokenizer 以便模型内初始化
        dropout_p=args.dropout
    )
    model = model.to(args.device)
    

    # soft prompt token IDs
    soft_prompt_ids = tokenizer.convert_tokens_to_ids(soft_prompt_tokens)  # [soft_A1]-[soft_A5] 和 [soft_B1]-[soft_B5]
    print('soft_prompt_ids',soft_prompt_ids)


    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': args.bert_lr},
        {'params': model.multi_label_classifier.parameters(), 'lr': args.multi_classifier_lr},
        {'params': model.single_label_classifier.parameters(), 'lr': args.single_classifier_lr},
      
    ])


    scheduler = None

    os.makedirs(args.save_dir, exist_ok=True)


    if args.loss_type == 'weighted_bce' and pos_weight_multi is not None:
        model.multi_label_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_multi.to(args.device))
    else:
        model.multi_label_loss_fn = nn.BCEWithLogitsLoss()

    model.single_label_loss_fn = nn.CrossEntropyLoss()


    train_and_evaluate(model, train_data_loader, dev_data_loader, optimizer, scheduler, device=args.device, args=args)
    

def get_eval_data(args):

    tokenizer = BertTokenizer.from_pretrained(args.model_name)


    # [CLS_A] 和 [soft_A1]-[soft_A5] 分别用于多标签任务
    # [CLS_B] 和 [soft_B1]-[soft_B5] 分别用于单标签任务
    soft_prompt_tokens = [ '[soft_A1]', '[soft_A2]', '[soft_A3]', '[soft_A4]', '[soft_A5]','[CLS_A]',
                           '[soft_B1]', '[soft_B2]', '[soft_B3]', '[soft_B4]', '[soft_B5]','[CLS_B]']
    print("Soft prompt tokens:", soft_prompt_tokens)


    tokenizer.add_tokens(soft_prompt_tokens, special_tokens=True)


    train_texts, train_label_lists, train_single_labels = load_data(args.train_data_dir)
    dev_texts, dev_label_lists, dev_single_labels = load_data(args.dev_data_dir)


    print('过采样前', len(train_texts))
    train_texts, train_label_lists, train_single_labels = oversample_data(train_texts, train_label_lists, train_single_labels, min_count=10)
    print('过采样后', len(train_texts))


    if args.loss_type == 'weighted_bce':
        pos_weight_multi = compute_pos_weight(train_label_lists)
    else:
        pos_weight_multi = None


    train_dataset = TextDataset(train_texts, train_label_lists, train_single_labels, tokenizer, args.max_length, soft_prompt_tokens)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(train_dataset[0])

    dev_dataset = TextDataset(dev_texts, dev_label_lists, dev_single_labels, tokenizer, args.max_length, soft_prompt_tokens)
    dev_data_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    print(dev_dataset[0])

    n_multi_labels = len(train_label_lists[0])
    n_single_labels = n_multi_labels  # 假设单标签是整数编码
    print(f"Number of multi-labels: {n_multi_labels}")
    print(f"Number of single-labels: {n_single_labels}")


    model = MultiTaskTextClassifier(
        model_name=args.model_name,
        n_multi_labels=n_multi_labels,
        n_single_labels=n_single_labels,
        prompt_tokens=soft_prompt_tokens,
        tokenizer=tokenizer,  # 传递 tokenizer 以便模型内初始化
        dropout_p=args.dropout
    )
    model = model.to(args.device)
    

    #soft prompt token IDs
    soft_prompt_ids = tokenizer.convert_tokens_to_ids(soft_prompt_tokens)  # [soft_A1]-[soft_A5] 和 [soft_B1]-[soft_B5]
    print('soft_prompt_ids',soft_prompt_ids)


    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': args.bert_lr},
        {'params': model.multi_label_classifier.parameters(), 'lr': args.multi_classifier_lr},
        {'params': model.single_label_classifier.parameters(), 'lr': args.single_classifier_lr},
      
    ])


    scheduler = None

    os.makedirs(args.save_dir, exist_ok=True)


    if args.loss_type == 'weighted_bce' and pos_weight_multi is not None:
        model.multi_label_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_multi.to(args.device))
    else:
        model.multi_label_loss_fn = nn.BCEWithLogitsLoss()

    model.single_label_loss_fn = nn.CrossEntropyLoss()


    eval_model(model, dev_loader, device=args.device)

if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Multi-Task Text Classification Model Training")
    parser.add_argument('--model_name', type=str, default='./model/', help='Name or path of the pre-trained model')
    parser.add_argument('--max_length', type=int, default=64, help='Maximum input length for the Tokenizer')
    parser.add_argument('--batch_size', type=int, default=300, help='Training batch size')
    parser.add_argument('--num_neg_samples', type=int, default=50,
                        help='Number of negative labels to randomly select within a batch')
    parser.add_argument('--loss_weight', type=float, default=0.1,
                        help='Learning rate for the BERT encoder (note: duplicate parameter name, adjust if needed)')
    parser.add_argument('--bert_lr', type=float, default=1e-5, help='Learning rate for the BERT encoder')
    parser.add_argument('--multi_classifier_lr', type=float, default=1e-3,
                        help='Learning rate for the multi-label classifier')
    parser.add_argument('--single_classifier_lr', type=float, default=1e-3,
                        help='Learning rate for the single-label classifier')
    parser.add_argument('--prompt_lr', type=float, default=5e-5, help='Learning rate for soft prompts')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Training device (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='./output/', help='Directory to save the best model')
    parser.add_argument('--train_data_dir', type=str, default='./data/train_label_data.json',
                        help='File path of the JSON file for training data')
    parser.add_argument('--dev_data_dir', type=str, default='./data/dev_label_data.json',
                        help='File path of the JSON file for development/validation data')
    parser.add_argument('--label_data_dir', type=str, default='./data/unique_labels.json',
                        help='File path of the JSON file for unique labels')
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'weighted_bce'],
                        help='Loss function type: bce | weighted_bce (Binary Cross-Entropy / Weighted Binary Cross-Entropy)')

    parser.add_argument('--soft_prompt_length', type=int, default=5, help='Length of soft prompts for each task')
    parser.add_argument('--eval_steps', type=int, default=25,
                        help='Number of steps between evaluations (evaluate once every N steps)')

    args = parser.parse_args()


    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
