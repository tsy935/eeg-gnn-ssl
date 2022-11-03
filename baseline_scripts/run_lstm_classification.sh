#!/bin/bash

INPUT_DIR=<resampled-dir>
RAW_DATA_DIR=<tusz-data-dir>
SAVE_DIR=<save-dir>

cd ..
python train.py \
    --model_name 'lstm' \
    --input_dir $INPUT_DIR \
    --raw_data_dir $RAW_DATA_DIR \
    --save_dir $SAVE_DIR \
    --max_seq_len 60 \
    --do_train \
    --num_epochs 60 \
    --task "classification" \
    --metric_name "F1" \
    --use_fft \
    --lr_init 3e-4 \
    --num_rnn_layers 2 \
    --rnn_units 64 \
    --num_classes 4 \
    --data_augment
