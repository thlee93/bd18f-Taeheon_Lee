#!/bin/bash

PARALL_HOME=$1

TRAIN1=$2
TRAIN2=$3

TENSORS="./list_tensor.txt"

OUTPUT1="./output1"
OUTPUT2="./output2"

CKPTS1=$4
CKPTS2=$5

COMPARE="compare_${TRAIN1}_${TRAIN2}"
echo > ${COMPARE}

while read -r line
do
    python ${PARALL_HOME}/tensorflow/tensorflow/python/tools/inspect_checkpoint.py --file_name=${TRAIN1}/model.ckpt-${CKPTS1} --tensor_name=${line} > $OUTPUT1
    python ${PARALL_HOME}/tensorflow/tensorflow/python/tools/inspect_checkpoint.py --file_name=${TRAIN2}/model.ckpt-${CKPTS2} --tensor_name=${line} > $OUTPUT2
    diff ${OUTPUT1} ${OUTPUT2} >> ${COMPARE}
done < $TENSORS
