#!/bin/bash

{
    curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
    # Change it when necessary
    # source activate pt

    train_id="RN20_test"
    train_stamp="$(date +"%Y%m%d_%H%M%S")"
    train_id=${train_id}_${train_stamp}

    result_dir="$curr_dir/results/$train_id"
    mkdir -p $result_dir

    python3 -u trunc_train.py \
        --model resnet20q \
        --dataset cifar10 \
        --train_split train \
        --lr 0.001 \
        --lr_decay "50,150,180" \
        --epochs 100 \
        --batch-size 128 \
        --optimizer adam \
        --weight-decay 0.0 \
        --results-dir $result_dir \
        --bit_width_list "1,2,3,4,5,6,7,8" \
        --is_training "F" \
        --truncation "F"
} && exit
