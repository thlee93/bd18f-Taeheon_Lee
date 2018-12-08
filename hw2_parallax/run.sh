#!/bin/bash

LOGDIR='./logs/single-gpu'
if [ -d ${LOGDIR} ]
then 
    rm -r ${LOGDIR}
fi

mkdir -p ${LOGDIR}
mkdir "${LOGDIR}/test"
mkdir "${LOGDIR}/train"

python run.py --num_windows 256 256 256 256 256 256 256 256 \
              --window_lengths 8 12 16 20 24 28 32 36 \
              --num_hidden 2000 \
              --batch_size 100 \
              --keep_prob 0.7 \
              --learning_rate 0.001 \
              --regularizer 0.001 \
              --max_epoch 25 \
              --seq_len 1000 \
              --num_classes 1000 \
              --log_interval 100 \
              --save_interval 100 \
              --test_file './data/test.txt' \
              --train_file './data/train.txt' \
              --log_dir ${LOGDIR} \
              --checkpoint_path ${LOGDIR}'/save'
