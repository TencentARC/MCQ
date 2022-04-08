#!/usr/bin/env bash
NOW="`date +%Y%m%d%H%M%S`"
JOB_NAME=MCQ_1frame
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=2628 train_dist_MCQ.py \
--config configs/dist-1frame-mask-noun.json --launcher pytorch \
