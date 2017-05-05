#!/bin/bash
. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

gpuid=9
export CUDA_VISIBLE_DEVICES=$gpuid
python3 main.py