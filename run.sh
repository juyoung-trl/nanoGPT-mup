#!/bin/bash

# 1. Prepare the data

python data/openwebtext/prepare.py


# 2. Train the model

torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
