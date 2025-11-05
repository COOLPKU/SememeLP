#!/usr/bin/env bash

set -x
set -e

model_path="./checkpoint/model_best.mdl"
task="WN18RR"
test_path="./data/${task}/test.txt.json"


neighbor_weight=0.05
rerank_n_hop=5



python3 -u evaluate.py \
--task "${task}" \
--is-test \
--eval-model-path "${model_path}" \
--neighbor-weight "${neighbor_weight}" \
--rerank-n-hop "${rerank_n_hop}" \
--train-path "./data/${task}/train.txt.json" \
--valid-path "${test_path}" "$@"
