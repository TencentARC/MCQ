#!/usr/bin/env bash
NOW="`date +%Y%m%d%H%M%S`"
JOB_NAME=test_retrieval

python test_MILES.py \
--config configs/zero_msrvtt_4f_i21k_MILES.json \

