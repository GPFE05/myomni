#!/bin/bash

# Qwen1.5-MoE-A2.7B W4A16 quantization script
# For PPL evaluation only (no quantization training)

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model Qwen/Qwen1.5-MoE-A2.7B \
    --net Qwen1.5-MoE-A2.7B \
    --eval_ppl \
    --wbits 4 \
    --abits 16 \
    --group_size 128 \
    --lwc \
    --epochs 0
