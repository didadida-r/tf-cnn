#!/bin/bash
if ! [ $# -eq 1 ];then
  echo "usage : $0 <num_require_gpu>
     num_gpu : Number of gpu to require."
  exit 1;
fi
require_gpu=$1

if ! nvidia-smi; then
  echo "$(hostname) can not use GPU"
  # exit 0;
else

# gpuid=`nvidia-smi --query-gpu=index,memory.used,utilization.gpu  --format=csv | grep "0 MiB, 0 %" | cut -f1 -d',' | head -n $require_gpu`
gpuid=`nvidia-smi --query-gpu=index,memory.used,utilization.gpu  --format=csv | grep "0 %" | cut -f1 -d',' | head -n $require_gpu`

if [ -z $gpuid ];then
	echo "Your requirement can not satisfy"
	nvidia-smi --query-gpu=index,utilization.gpu,memory.used  --format=csv 
	exit 1;
fi
gpuid=`echo $gpuid | tr ' ' ','`
export CUDA_VISIBLE_DEVICES=$gpuid
echo "export CUDA_VISIBLE_DEVICES=$gpuid" 

fi

