#!/usr/bin/env bash
CONFIG=$1
PORT_=$2
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29500 basicsr/train.py -opt Allweather/Options/Allweather_Histoformer.yml --launcher pytorch