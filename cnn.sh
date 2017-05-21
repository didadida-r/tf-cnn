#!/bin/bash
. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

gpuid=0
export CUDA_VISIBLE_DEVICES=$gpuid
python3 -u main.py