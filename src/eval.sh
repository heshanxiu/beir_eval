#! /usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 nohup python -u eval_beir.py \
      //mnt/efs/fs1/sbert/examples/training/ms_marco/output/train_bi-encoder-mnrl-.-output-train_bi-encoder-mnrl-distilbert-base-uncased-margin_3.0-2022-11-15_00-55-03-31200-margin_3.0-2022-11-15_04-27-39/124800 \
      miniLM6_mnrl_dual_beir.json > miniLM6_dual_mnrl_beir.txt 2>&1 &