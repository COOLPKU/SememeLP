#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="./checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="./data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model "./model" \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "./data/${TASK}/train.txt.json" \
--valid-path "./data/${TASK}/valid.txt.json" \
--task ${TASK} \
--batch-size 128 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--pre-batch 3 \
--finetune-t \
--epochs 20 \
--workers 1 \
--max-to-keep 3 "$@"
