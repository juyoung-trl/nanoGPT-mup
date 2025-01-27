#!/bin/bash

export WANDB_API_KEY=19cb221517afaa1880927f5bb01888f9c319d444
export CUDA_VISIBLE_DEVICES=0

# Single-GPU Launching
LAUNCHER=python

# Multi-GPU Launching (single node)
GPU=8
# LAUNCHER="torchrun --standalone --nproc_per_node=$GPU"
TRAIN_SCRIPT="/mnt/home/juyoung.suk/nanoGPT-mup/train.py"

LAYERS=2

for width in 256 512 1024 2048; do
    for lr in 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625 0.0001220703125 0.00006103515625; do
        for seed in 1 2 3; do
            head_size=64
            n_heads=$((width / head_size))
            min_lr=$(awk "BEGIN {print $lr/10}")
            out_dir="mup_examples/mutransfer_lr_owt/sp/out/width${width}_depth${LAYERS}_seed${seed}_lr${lr}"
            $LAUNCHER $TRAIN_SCRIPT \
                --out_dir=$out_dir \
                --eval_interval=1 \
                --log_interval=1 \
                --eval_iters=1 \
                --eval_only=False \
                --skip_val_loss=False \
                --always_save_checkpoint=False \
                --never_save_checkpoint=True \
                --init_from='scratch' \
                --wandb_log=True \
                --wandb_project='mup' \
                --wandb_run_name="mutransfer_lr_owt_sp_width${width}_depth${LAYERS}_seed${seed}_lr${lr}" \
                --csv_log=True \
                --dataset='openwebtext' \
                --gradient_accumulation_steps=1 \
                --batch_size=32 \
                --block_size=1024 \
                --n_layer=2 \
                --n_head=$n_heads \
                --n_embd=$width \
                --dropout=0.0 \
                --bias=False \
                --init_std=0.02 \
                --learning_rate=$lr \
                --lr_decay_iters=1000 \
                --min_lr=$min_lr \
                --max_iters=1000 \
                --weight_decay=1e-1 \
                --beta1=0.9 \
                --beta2=0.95 \
                --grad_clip=1.0 \
                --decay_lr=True \
                --seed=$seed \
                --backend='nccl' \
                --device='cuda' \
                --dtype='bfloat16' \
                --compile=True
        done
    done
done
