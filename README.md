# How Sememic Components Can Benefit Link Prediction for Lexico-Semantic Knowledge Graphs?
[![Paper](https://img.shields.io/badge/Paper-EMNLP%202025-blue)](https://arxiv.org/abs/XXX)

Data and code for the paper "How Sememic Components Can Benefit Link Prediction for Lexico-Semantic Knowledge Graphs?" in EMNLP 2025 Main.

## 📊 Data
We provide the SememeDef dataset for Sememe Prediction (SP), along with two Chinese datasets, HN7 and CWN5, for Link Prediction (LP), aiming to alleviate the scarcity of both SP and Chinese LP resources.

### 1. SememeDef (Sememe Prediction Dataset)
#### General description
Our SememeDef dataset contains a substantial amount of word senses with sememe annotations for both English and Chinese (70,645 English samples and 43,163 Chinese ones, covering 2,042 and 1,762 sememes, respectively). Among them, the English data is integrated from [SememeBabel](https://github.com/thunlp/MSGI), while the Chinese data is obtained based on Word Sense Alignment (WSA) results between Hownet and CCD.

To respect intellectual property rights and better meet the needs of computational applications, we have referred to the approach of [MiCLS](https://github.com/COOLPKU/MorBERT), condensed the relevant semantic space to some extent, and made extensive revisions and optimizations to the word sense definitions.

Here, we have open-sourced a subset of the Chinese part of this resource, which includes 20,492 word senses for 15,000 words (the vocabulary is consistent with the [MiCLS](https://github.com/COOLPKU/MorBERT) resource previously open-sourced by [COOLPKU](https://github.com/COOLPKU/COOL). As a result, the experimental results for Chinese in the paper may be slightly affected during replication.

These resources are uploaded to `data/SememeDef/`. The file structure is as follows:
- `data/SememeDef/`: data files
  - `en.txt`: Data for English
  - `zh.txt`: Data for Chinese

#### Data Format
Each sample includes a word sense definition, corresponding sememes, and the main sememe. The columns are separated by \\t. The format and example of the data are shown in the table:
|Column|Description|Example|
|:----|:----|:----|
|**0**|Word Sense Definition|a person whose job is teaching|
|**1**|Sememes|human\|人, occupation\|职位, education\|教育, teach\|教|
|**2**|MainSememe|human\|人|

### 2. HN7 (Chinese LP Dataset)
#### General description
HN7 is built upon HowNet, following the instructions of [OpenHonwet](https://github.com/thunlp/SCPapers/blob/master/resources/2003_%E7%9F%A5%E7%BD%91.pdf). It also follows the [WN18RR](https://ojs.aaai.org/index.php/AAAI/article/view/11573) benchmark to avoid test leakage. Dataset Details are shown in the following table:

| Item                | Value                  |
|---------------------|------------------------|
| Synsets Count       | 10,939                 |
| Relation Types      | 7 (antonymy, hypernymy, hyponymy, holonymy, meronymy, material, product) |
| Split Ratio (Train:Val:Test) | 8:1:1 |
| Train Triples       | 13,624                 |
| Validation Triples  | 1,702                  |
| Test Triples        | 1,703                  |

These resources are uploaded to `data/HN7/`. The file structure is as follows:
- `data/HN7/`: data files
  - `synsets_data.txt`: Data for synsets in HN7, 
  - `train.tsv`: Data for the Train set,
  - `valid.tsv`: Data for the Valid set,
  - `test.tsv`: Data for the Test set,

#### Data Format
The file `synsets_data.txt` contains the information of synsets in HN7. The columns are separated by \\t. The format and example of the data are shown in the table:
|Column|Description|Example|
|:----|:----|:----|
|**0**|Synset ID|hw\_synset\_7451|
|**1**|Synset|\{戏剧性、戏剧化\}|
|**2**|Definition|形容词，引人注目的表现或变化。|

The files `train.tsv`, `val.tsv` and `test.tsv` contain the triples in HN7. The columns are separated by \\t. The format and example of the data are shown in the table:
|Column|Description|Example|
|:----|:----|:----|
|**0**|Head|hw\_synset\_7451|
|**1**|Relation|反义词|
|**2**|Tail|hw\_synset\_175|


### 3. CWN5 (Chinese LP Dataset)
#### Overview
CWN5 is derived from [Chinese WordNet](https://lopentu.github.io/CwnWeb/#:~:text=%E4%B8%AD%E6%96%87%E8%A9%9E%E5%BD%99%E7%B6%B2%E8%B7%AF%EF%BC%88Chinese%20Wordnet%EF%BC%8CCWN%EF%BC%89%EF%BC%8C%E6%98%AF%E4%B8%80%E9%A0%85%E8%A9%A6%E5%9C%96%E8%A7%A3%E6%B1%BA%E8%A9%9E%E7%BE%A9%EF%BC%88sense%EF%BC%89%E4%BB%A5%E5%8F%8A%E8%A9%9E%E5%BD%99%E8%AA%9E%E6%84%8F%E9%97%9C%E4%BF%82%EF%BC%88lexical,semantic%20relations%EF%BC%89%E7%9A%84%E8%AA%9E%E8%A8%80%E7%9F%A5%E8%AD%98%E8%B3%87%E6%BA%90%E3%80%82%20%E4%B8%AD%E6%96%87%E8%A9%9E%E7%B6%B2%E7%9A%84%E6%A0%B8%E5%BF%83%E5%85%83%E7%B4%A0%E6%98%AF%E4%B8%AD%E6%96%87%E8%A9%9E%E5%BD%99%E7%9A%84%E5%90%8C%E7%BE%A9%E8%A9%9E%E9%9B%86%EF%BC%88synsets%EF%BC%89%E4%BB%A5%E5%8F%8A%E9%80%A3%E7%B9%AB%E5%90%84%E8%A9%9E%E9%9B%86%E7%9A%84%E8%AA%9E%E6%84%8F%E9%97%9C%E4%BF%82%EF%BC%9B%E9%80%8F%E9%81%8E%E8%AA%9E%E6%84%8F%E9%97%9C%E4%BF%82%EF%BC%8C%E5%B0%87%E5%90%84%E5%80%8B%E5%90%8C%E7%BE%A9%E8%A9%9E%E9%9B%86%E9%80%A3%E6%8E%A5%E8%B5%B7%E4%BE%86%EF%BC%8C%E5%BD%A2%E6%88%90%E8%AA%9E%E6%84%8F%E7%B6%B2%E7%B5%A1%E3%80%82) (CWN), converted to simplified Chinese. It also follows the [WN18RR](https://ojs.aaai.org/index.php/AAAI/article/view/11573) benchmark to avoid test leakage.

#### Dataset Details
| Item                | Value                  |
|---------------------|------------------------|
| Synsets Count       | 3,149                  |
| Relation Types      | 5 (antonymy, hypernymy, hyponymy, holonymy, meronymy) |
| Split Ratio (Train:Val:Test) | 8:1:1 |
| Train Triples       | 2,600                  |
| Validation Triples  | 324                    |
| Test Triples        | 325                    |

These resources are uploaded to `data/CWN5/`. The file structure is as follows:
- `data/CWN5/`: data files
  - `synsets_data.txt`: Data for synsets in CWN5, 
  - `train.tsv`: Data for the Train set,
  - `valid.tsv`: Data for the Valid set,
  - `test.tsv`: Data for the Test set,

#### Data Format
The file `entities_data.txt` contains the information of synsets in CWN5. The columns are separated by \\t. The format and example of the data are shown in the table:
|Column|Description|Example|
|:----|:----|:----|
|**0**|Synset ID|syn\_004810|
|**1**|Synset|\{交通工具\}|
|**2**|Definition|名词，人类为了方便移动而搭乘或驾驶的工具。|


The files `train.tsv`, `val.tsv` and `test.tsv` contain the triples in CWN5. The columns are separated by \\t. The format and example of the data are shown in the table:
|Column|Description|Example|
|:----|:----|:----|
|**0**|Head|syn\_002445|
|**1**|Relation|上位词|
|**2**|Tail|syn\_002068|

## 🛠️ Code
### Installation
We recommend using `conda` to manage the environment.

1. Create and activate the environment:
```bash
conda create -n sememelp python==3.10
conda activate sememelp
```

2. Clone the repository and install dependencies:
```bash
git clone https://github.com/[your-username]/SememeLP.git
cd SememeLP
pip3 install -e .
# Install additional dependencies (to be filled by user, e.g., torch, transformers)
# pip3 install torch==2.1.0 transformers==4.35.2 ...
```

3. Install optional dependencies (if needed):
```bash
# For GPU acceleration (example)
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
# For FlashAttention (optional)
pip3 install flash-attn --no-build-isolation
```

### Quick Start
#### 1. Data Preparation
- Download the datasets from [to be filled by user, e.g., Google Drive/Weiyun] and extract them to the `data/` directory.
- The directory structure should be:
```
SememeLP/
├── data/
│   ├── sememedef/
│   │   ├── train/
│   │   └── val/
│   ├── hn7/
│   │   ├── train_triples.tsv
│   │   ├── val_triples.tsv
│   │   ├── test_triples.tsv
│   │   └── definitions.json
│   └── cwn5/
│       ├── train_triples.tsv
│       ├── val_triples.tsv
│       ├── test_triples.tsv
│       └── definitions.json
└── src/
```

#### 2. Training
We provide training scripts for two core stages: (1) Sememe Encoder Fine-tuning (SP task), (2) SememeLP Training (LP task).

##### Stage 1: Fine-tune Sememe Encoder
```bash
cd scripts/train
bash train_sememe_encoder.sh \
    --train_data ../data/sememedef/train \
    --val_data ../data/sememedef/val \
    --model_name [base-model-name, e.g., chinese-bert-base-wwm-ext] \
    --output_dir ../checkpoints/sememe_encoder \
    # Additional hyperparameters (to be filled by user)
```

##### Stage 2: Train SememeLP
```bash
cd scripts/train
bash train_sememelp.sh \
    --dataset [hn7/cwn5/wn18rr] \
    --data_dir ../data/[hn7/cwn5] \
    --sememe_encoder_path ../checkpoints/sememe_encoder/best_model \
    --output_dir ../checkpoints/sememelp \
    # Additional hyperparameters (to be filled by user, e.g., batch_size, epochs)
```

#### 3. Inference
Use the trained SememeLP model for link prediction inference:
```bash
cd scripts/inference
python run_inference.py \
    --model_path ../checkpoints/sememelp/best_model \
    --dataset [hn7/cwn5] \
    --test_data ../data/[hn7/cwn5]/test_triples.tsv \
    --output_path ../results/inference_output.json \
    # Additional parameters (to be filled by user)
```

#### 4. Evaluation
Evaluate the model performance using MRR, Hits@1/3/10 metrics:
```bash
cd scripts/evaluation
python run_eval.py \
    --pred_path ../results/inference_output.json \
    --ground_truth_path ../data/[hn7/cwn5]/test_triples.tsv \
    --metric [mrr/hits@1/hits@3/hits@10] \
    --output_path ../results/eval_results.txt \
    # Additional parameters (to be filled by user)
```

## 📈 Results
SememeLP achieves SOTA performance on English and Chinese LP datasets:
| Dataset  | MRR   | Hits@1 | Hits@3 | Hits@10 |
|----------|-------|--------|--------|---------|
| WN18RR   | 75.1% | 67.6%  | 79.8%  | 88.5%   |
| HN7      | 80.5% | 74.6%  | 84.0%  | 91.8%   |
| CWN5     | 77.1% | 69.2%  | 82.5%  | 90.6%   |

For detailed results and analysis, refer to our paper.

## 🤝 Acknowledgements
This work is based on the following open-source projects:
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [HowNet](https://www.keenage.com/)
- [Chinese WordNet](https://lope.linguistics.ntu.edu.tw/cwn/)

We sincerely appreciate their contributions to the community.

## 📚 Citation
If you find this work useful, please cite our paper:
```bibtex
@inproceedings{sememelp2025,
  title={How Sememic Components Can Benefit Link Prediction for Lexico-Semantic Knowledge Graphs?},
  author={[Author Names]},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2025}
}
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

要不要我帮你补充**数据集下载链接模板**或**代码占位符的具体示例**？比如细化训练脚本的参数说明，或补充数据集的样例文件内容。
