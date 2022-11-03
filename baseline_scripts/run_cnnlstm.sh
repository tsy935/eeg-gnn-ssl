#!/bin/bash

INPUT_DIR=<resampled-dir>
RAW_DATA_DIR=<tusz-data-dir>
SAVE_DIR=<save-dir>

cd ..
python train.py \
    --model_name 'cnnlstm' \
    --input_dir $INPUT_DIR \
    --raw_data_dir $RAW_DATA_DIR \
    --save_dir $SAVE_DIR \
    --max_seq_len 60 \
    --do_train \
    --num_epochs 100 \
    --task "detection" \
    --metric_name "auroc" \
    --use_fft \
    --lr_init 1e-4 \
    --num_classes 1 \
    --data_augment
