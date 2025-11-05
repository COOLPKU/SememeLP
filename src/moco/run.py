import ast
import torch
import argparse
from transformers import logging, BertTokenizer
import time
import os
import pandas as pd
from utils import Entities, Relations, Konwledge_Graph, set_seed, collate
from MoCoKGC import MoCoKGC
from trainer import Trainer
from evaluator import Evaluator
import json
from load_eval_sp_model import load_hw_main
def _setup_training(model, device):
    if torch.cuda.device_count() > 1 and False:
        model.cuda()
        model.hr_encoder = torch.nn.DataParallel(model.hr_encoder).cuda()
        model.e_encoder = torch.nn.DataParallel(model.e_encoder).cuda()
        model.momentum_e_encoder = torch.nn.DataParallel(model.momentum_e_encoder).cuda()
    elif torch.cuda.is_available():
        model.to(device)
        # for name, module in model.named_children():
        #     if name != 'E_emb':
        #         module.cuda()
    else:
        print('No gpu will be used')
    return model

def main(args):
    set_seed(args.seed)
    logging.set_verbosity_error()
    tokenizer = BertTokenizer.from_pretrained(args.plm_name, local_files_only=True)

    print('加载hw_model........')
    hw_tokenizer,hw_model = load_hw_main()
    print('已经加载完成hw_model。')


    entity_tokenizer = tokenizer # tokenizer, None
    print("加载Entities")
    entities = Entities(args, args.data_dir, tokenizer=entity_tokenizer,hw_tokenizer=hw_tokenizer)
    print("加载Relations")
    relations = Relations(args, args.data_dir)
    # datasets
    train_data = Konwledge_Graph(args, data_dir=args.data_dir, data_type = "train", entities=entities, relations=relations, add_neighbor_name=args.add_neighbor_name, tokenizer=tokenizer,hw_tokenizer=hw_tokenizer)
    valid_data = Konwledge_Graph(args, data_dir=args.data_dir, data_type = "valid", reverse=False, KG=train_data, add_hr_vocab=True,hw_tokenizer=hw_tokenizer)
    valid_reverse_data = Konwledge_Graph(args, data_dir=args.data_dir, data_type = "valid", reverse=True, KG=train_data, add_hr_vocab=True,hw_tokenizer=hw_tokenizer)
    test_data = Konwledge_Graph(args, data_dir=args.data_dir, data_type = "test", reverse=False, KG=train_data, add_hr_vocab=True,hw_tokenizer=hw_tokenizer)
    test_reverse_data = Konwledge_Graph(args, data_dir=args.data_dir, data_type = "test", reverse=True, KG=train_data, add_hr_vocab=True,hw_tokenizer=hw_tokenizer)
    if not (args.data_dir=="./data/Wikidata5M/wikidata5m_inductive/" or args.data_dir=="./data/Wikidata5M/wikidata5m_transductive/"):
        test_train_data = Konwledge_Graph(args, data_dir=args.data_dir, data_type = "train", reverse=False, KG=train_data, add_hr_vocab=False,hw_tokenizer=hw_tokenizer)
        test_train_reverse_data = Konwledge_Graph(args, data_dir=args.data_dir, data_type = "train", reverse=True, KG=train_data, add_hr_vocab=False,hw_tokenizer=hw_tokenizer)
    # modeling
    print("modeling")
    num_training_steps = args.epochs * len(train_data) // max(args.batch_size, 1)
    model = MoCoKGC(args, len(entities), len(relations)*2, NeighborsRepresentationSets=train_data.entity_neighborhood, num_training_steps=num_training_steps)
    if args.checkpoint_path is not None:
        state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        
    else:
        state_dict = None
    _setup_training(model, args.device)
    model.set_E_emb(state_dict)

    # training and testing
    if not (args.data_dir=="./data/Wikidata5M/wikidata5m_inductive/" or args.data_dir=="./data/Wikidata5M/wikidata5m_transductive/"):
        evaluator = Evaluator(args, entities, valid_data, valid_reverse_data, test_data, test_reverse_data, test_train_data, test_train_reverse_data)
    else:
        evaluator = Evaluator(args, entities, valid_data, valid_reverse_data, test_data, test_reverse_data)
    if args.task=="train":
        trainer = Trainer(args, model, hw_model, evaluator, entities, train_data)
        trainer.run()
    elif args.task=="test":
        model.eval()
        # metric = evaluator.train(model)
        # print(metric)
        # metric = evaluator.valid(model)
        # print(metric)
        # model.update_E_enc(evaluator.entities_loader)
        # model.update_E_enc(evaluator.entities_loader)
        # model.update_E_enc(evaluator.entities_loader)
        for i in range(10):
            metric = evaluator.test(model,hw_model, updata_E_enc=False, inductive=True if args.data_dir=="./data/Wikidata5M/wikidata5m_inductive/" else False)
            print(metric)
    elif args.task=="pretrain":
        trainer = Trainer(args, model, evaluator, entities, train_data)
        trainer.pretrain()
        

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/WN18RR/", nargs="?",
                    help="Which dataset to use: FB15k-237, WN18RR or NELL.")
    parser.add_argument("--device", type=str, default="cuda:0", nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--output_path", type=str, default='./outputs/', nargs="?",
                    help="output file path.")
    parser.add_argument("--AMP_enabled", type=ast.literal_eval, default=True, nargs="?",
                    help="AMP enabled.")
    parser.add_argument("--log_file", type=str, default="demo.log", nargs="?",
                    help="the path of log file.")
    parser.add_argument("--task", type=str, default=None, nargs="?",
                    help="Which task to use: pretrain, train or test.")
    parser.add_argument("--checkpoint_path", type=str, default='./outputs/WN18RR-r-ab/only_main/best.pt', nargs="?",
                    help="checkpoint_path")
    parser.add_argument("--num_workers", type=int, default=1, nargs="?",
                    help="num_workers")

                    

    parser.add_argument("--epochs", type=int, default=20, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=1024, nargs="?",
                    help="Batch size.")
    parser.add_argument("--actual_batch_size", type=int, default=64, nargs="?",
                    help="num_split.")
    parser.add_argument("--test_batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, nargs="?",
                    help="weight_decay.")
    parser.add_argument("--warmup", type=int, default=400, nargs="?",
                    help="warmup.")
    parser.add_argument("--margin", type=float, default=0.02, nargs="?",
                    help="margin.")
    parser.add_argument("--tau", type=float, default=0.05, nargs="?",
                    help="tau.")
    parser.add_argument("--finetune_t", type=ast.literal_eval, default=True, nargs="?",
                    help="finetune_t.")
    parser.add_argument("--seed", type=int, default=1024, nargs="?",
                    help="Random seed.")
    parser.add_argument("--eps", type=float, default=1e-12, nargs="?",
                    help="eps")
    parser.add_argument("--add_neighbor_name", type=ast.literal_eval, default=False, nargs="?",
                    help="add_neighbor_name")
    parser.add_argument("--save_intermediate_files", type=ast.literal_eval, default=False, nargs="?",
                    help="save_intermediate_files")

    parser.add_argument("--plm_name", type=str, default="/home/lqy/My_code/checkpoints/LMs/bert-base-uncased", nargs="?",
                    help="plm_name.")
    parser.add_argument("--e_max_length", type=int, default=32, nargs="?",
                    help="e_max_length.")
    parser.add_argument("--hr_max_length", type=int, default=32, nargs="?",
                    help="hr_max_length.")
    parser.add_argument("--r_prompt_len", type=int, default=4, nargs="?",
                    help="r_prompt_len.")
    parser.add_argument("--soft_prompt_len", type=int, default=4, nargs="?",
                    help="r_prompt_len.")
    parser.add_argument("--neighborhood_sample_K", type=int, default=5, nargs="?",
                    help="neighborhood_sample_K.")
    parser.add_argument("--extra_negative_sample_size", type=int, default=4096, nargs="?",
                    help="extra_negative_sample_size.")
    parser.add_argument("--emb_dropout", type=float, default=0.1, nargs="?",
                    help="emb_dropout.")
    parser.add_argument("--queue_size", type=int, default=8, nargs="?",
                    help="queue_size.")
    parser.add_argument("--m_decay", type=float, default=0.999, nargs="?",
                    help="m_decay.")
    parser.add_argument("--output_type", type=str, default="CLS", nargs="?",
                    help="output_type.")
    parser.add_argument("--entity_embedding_method", type=str, default="MLP", nargs="?",
                    help="entity_embedding_method, MLP,linear")
    
    parser.add_argument("--hr_neighborhood", type=ast.literal_eval, default=True, nargs="?",
                    help="hr_neighborhood")
    parser.add_argument("--e_neighborhood", type=ast.literal_eval, default=True, nargs="?",
                    help="e_neighborhood")
    parser.add_argument("--args_file_path", type=str, default='./outputs/WN18RR-r-ab/only_main/args.json', nargs="?",
                    help="args_file_path")
    parser.add_argument('--hw_feature_size', default=768, type=int, metavar='N',
                    help='')
                    

    args= parser.parse_args()
    # args.device = torch.device(args.device)
    args.cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    print(args)
    if args.task!="test":
        dataset_name = args.data_dir.split("/")[-2]
        output_dir = args.output_path + dataset_name
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        now_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = output_dir + "/" + now_time + "/"
        os.mkdir(output_dir)
        args.log_file = output_dir + args.log_file
        args.output_path = output_dir

        # 将args转换为字典并保存为JSON
        args.args_file_path = output_dir+'args.json'
        with open(args.args_file_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
    else:
        # 读取JSON文件并转换回字典
        assert args.args_file_path is not None, "args_file_path is not None"
        with open(args.args_file_path, 'r') as f:
            args_dict = json.load(f)
        args_dict.pop("task")
        args_dict.pop("checkpoint_path")
        # 使用字典更新args对象
        args.__dict__.update(args_dict)

    print(f"log_file: {args.log_file}")
    main(args)
                

