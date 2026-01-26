#!/bin/bash

# Qwen1.5-MoE-A2.7B W4A16 quantization script with LWC training

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model Qwen/Qwen1.5-MoE-A2.7B \
    --net Qwen1.5-MoE-A2.7B \
    --eval_ppl \
    --wbits 4 \
    --abits 16 \
    --group_size 128 \
    --lwc \
    --epochs 20 \
    --nsamples 128 \
    --output_dir ./log/Qwen1.5-MoE-A2.7B-w4a16 \
    --save_dir ./quantized_models/Qwen1.5-MoE-A2.7B-w4a16
