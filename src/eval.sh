#! /usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 nohup python -u eval_beir.py \
      /mnt/efs/fs1/sbert/examples/training/ms_marco/output/train_bi-encoder-mnrl-distilbert-base-uncased-margin_3.0-2022-11-09_23-32-10/156000 \
      miniLM6_mnrl_beir.json > miniLM6_mnrl_beir.txt 2>&1 &