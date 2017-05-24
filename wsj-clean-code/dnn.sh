#!/bin/bash
. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

# gpuid=0
# export CUDA_VISIBLE_DEVICES=$gpuid
. ./get_gpu.sh 1
python3 -u main-wsj.py