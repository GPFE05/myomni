#!/bin/bash
# Script 1: Baseline OmniQuant (No Gate Training)

CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model --model ./models/Qwen1.5-MoE-A2.7B \
    --net Qwen1.5-MoE-A2.7B \
    --eval_ppl \
    --wbits 4 \
    --abits 16 \
    --group_size 128 \
    --lwc \
    --epochs 0 \
    --batch_size 8 \
    --output_dir ./log/baseline_lwc \
    --wd 0