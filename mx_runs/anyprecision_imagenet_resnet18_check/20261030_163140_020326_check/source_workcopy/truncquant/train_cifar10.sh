#!/bin/bash

{
    curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
    # Change it when necessary
    source activate pt
    # export CUDA_VISIBLE_DEVICES=1,2

    train_id="RN20_truncready"
    train_stamp="$(date +"%Y%m%d_%H%M%S")"
    train_id=${train_id}_${train_stamp}

    result_dir="$curr_dir/results/$train_id"
    mkdir -p $result_dir

    python3 -u train.py \
        --model resnet20q \
        --dataset cifar10 \
        --train_split train \
        --lr 0.001 \
        --lr_decay "100,150,180" \
        --epochs 100 \
        --optimizer adam \
        --weight-decay 0.0 \
        --results-dir $result_dir \
        --bit_width_list "1,2,3,4,5,6,7,8" \
        --cal_bw "3,5,6,7" \
        --truncation "F" # true if you want to do trunc-ready training
} && exit
