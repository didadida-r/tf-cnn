#!/bin/bash
. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

feats_nj=10
stage=-2

# Config:
expdir=tf-exp/cnn          # the exp dir
logdir=$expdir/log
gmmdir=exp/tri3                 # the gmm dir
if [ ! -d $expdir ]; then
  mkdir -p $expdir
fi

Fbank_Feature=false
fMLLR_Feature=false

# Prepare feature
if $Fbank_Feature; then
  if [ $stage -le -5 ]; then
  echo ============================================================================
  echo "                                  Make FBank & Compute CMVN                    "
  echo ============================================================================

  # 这里需要将前面生成的data文件夹复制到data-fbank中，否则下面的程序会修改原来data中的数据格式为fbank
  # 选择不压缩
  srcdir=data-fbank   # 原始数据文件以及最终的scp
  fbankdir=fbank       # 中间生成文件以及生成的特征ark文件所在目录
  cp -r data $srcdir || exit 1;

  for x in train dev test; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 15 --compress false data-fbank/$x exp/make_fbank/$x $fbankdir
    steps/compute_cmvn_stats.sh data-fbank/$x exp/make_fbank/$x $fbankdir
  done

  fi

  # Prepare feature
  if [ $stage -le -4 ]; then
  echo ============================================================================
  echo "                                  Store fBank features                    "
  echo ============================================================================

  # store_fbank.sh 为自定义
  srcdir=data-fbank   # 原始数据文件以及最终的scp
  fbankdir=fbank       # 中间生成文件以及生成的特征ark文件所在目录
  for x in train dev test; do
    dir=$fbankdir/$x
    local/store_fbank.sh --nj $feats_nj --delta_order 2 --cmd "$train_cmd" \
      $dir $srcdir/$x $dir/log $dir/data || exit 1
  done

  fi
fi

if $fMLLR_Feature; then
  # Config:
  gmmdir=exp/tri3
  data_fmllr=data-fmllr-tri3

  # Store fMLLR features, so we can train on them easily,
  echo ============================================================================
  echo "                                  Store fMLLR features                    "
  echo ============================================================================
  # test
  dir=$data_fmllr/test
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_test \
     $dir data/test $gmmdir $dir/log $dir/data || exit 1
  # dev
  dir=$data_fmllr/dev
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_dev \
     $dir data/dev $gmmdir $dir/log $dir/data || exit 1
  # train
  dir=$data_fmllr/train
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali \
     $dir data/train $gmmdir $dir/log $dir/data || exit 1
fi

# if [ $stage -le -3 ];then
# echo ============================================================================
# echo "             Get the lda for the nnet training                            "
# echo ============================================================================
 
# local/get_lda.sh --cmd "$train_cmd" data-fmllr-tri3/train data/lang exp/tri3_ali $expdir
# fi


if [ $stage -le -2 ];then
echo ============================================================================
echo "             Train and decode neural network acoustic model               "
echo ============================================================================
aviliable_gpu_ids=1
CUDA_VISIBLE_DEVICES=$aviliable_gpu_ids python3 main.py
fi

exit

if [ $stage -le -1 ];then
echo ============================================================================
echo "                           Decode with language model                     "
echo ============================================================================
#copy the gmm model and some files to speaker mapping to the decoding dir
cp exp/tri3/final.mdl $expdir
cp -r exp/tri3/graph $expdir
for x in dev test;do
  if [ ! -d $expdir/decode_${x} ]; then 
    mkdir -p $expdir/decode_${x}
  fi
  cp data-fmllr-tri3/${x}/utt2spk $expdir/decode_${x}
  cp data-fmllr-tri3/${x}/text $expdir/decode_${x}
  cp data-fmllr-tri3/${x}/stm $expdir/decode_${x}
  cp data-fmllr-tri3/${x}/glm $expdir/decode_${x}

  #decode using kaldi
  local/kaldi/decode.sh --cmd "$train_cmd" --nj 10 $expdir/graph $expdir/decode_${x} $expdir/decode_${x} | tee $expdir/decode_${x}.log || exit 1;  
done
fi
