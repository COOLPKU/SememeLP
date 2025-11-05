#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0 python run.py \
    --data_dir ./data/WN18RR/ \
    --device cuda:0 \
    --AMP_enabled True \
    --task test \
    --epochs 30 \
    --warmup 10 \
    --batch_size 64 \
    --actual_batch_size 64 \
    --test_batch_size 256 \
    --finetune_t True \
    --tau 0.05 \
    --margin 0.02 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --neighborhood_sample_K 16\
    --r_prompt_len 4 \
    --extra_negative_sample_size 32 \
    --m_decay 0.999 \
    --add_neighbor_name True \
    --queue_size 300 \
    --e_max_length 64 \
    --hr_max_length 64 \
    --entity_embedding_method MLP \
    --hr_neighborhood True \
    --e_neighborhood True \
    --plm_name ./model/