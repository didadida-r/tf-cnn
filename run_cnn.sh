#!/bin/bash
. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

gpu_ids=4

feats_nj=20
stage=-2

Fbank_Feature=false
MFCC_Feature=true
fMLLR_Feature=false

# 这里需要将前面生成的data文件夹复制到data-fbank中，否则下面的程序会修改原来data中的数据格式为fbank
# 选择不压缩
desdir=data-tf       # 原始数据文件以及最终的scp
tmpdir=data-tmp       # 中间生成文件以及生成的特征ark文件所在目录

# 是否添加deltas特征
apply_deltas=false
    
# 生成fbank特征
if $Fbank_Feature; then

    echo ============================================================================
    echo "            Make FBank & Compute CMVN                    "
    echo ============================================================================

    # 删除原来的数据，并将data目录复制到desdir
    #rm -rf $desdir $tmpdir
    cp -r data $desdir || exit 1;

    for x in train dev test; do
        steps/make_fbank.sh --cmd "$train_cmd" --nj $feats_nj --compress false $desdir/$x exp/make_fbank/$x $tmpdir
        steps/compute_cmvn_stats.sh $desdir/$x exp/make_fbank/$x $tmpdir
    done
    
    echo ============================================================================
    echo "                                  Store fBank features                    "
    echo ============================================================================
    for x in train dev test; do
        dir=$tmpdir/$x
        local/store_fbank.sh --nj $feats_nj --apply_deltas $apply_deltas --cmd "$train_cmd" \
          $dir $desdir/$x $dir/log $dir/data || exit 1
    done
fi

# 重新生成mfcc并进行deltas变换
if $MFCC_Feature; then
    echo ============================================================================
    echo "         MFCC Feature Extration & CMVN           "
    echo ============================================================================

    rm -rf $desdir $tmpdir
    cp -r data $desdir || exit 1;

    for x in train dev test; do 
        steps/make_mfcc.sh --cmd "$train_cmd" --nj $feats_nj $desdir/$x exp/make_mfcc/$x $tmpdir
        steps/compute_cmvn_stats.sh $desdir/$x exp/make_mfcc/$x $tmpdir
    done
    

    echo ============================================================================
    echo "             Store mfcc with deltas and cmvn                    "
    echo ============================================================================

    for x in train dev test; do
        dir=$tmpdir/$x
        local/store_fbank.sh --nj $feats_nj --apply_deltas $apply_deltas --cmd "$train_cmd" \
          $dir $desdir/$x $dir/log $dir/data || exit 1
    done
fi

# 对于mfcc特征进行fmllr变换
if $fMLLR_Feature; then
  # Config:
  gmmdir=exp/tri3

  echo ============================================================================
  echo "                                  Store fMLLR features                    "
  echo ============================================================================
  for x in test dev; do
      dir=$desdir/$x
      steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
         --transform-dir $gmmdir/decode_$x \
         $dir data/$x $gmmdir $dir/log $dir/data || exit 1
  done
  # train
  dir=$desdir/train
  steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali \
     $dir data/train $gmmdir $dir/log $dir/data || exit 1
fi

exit

if [ $stage -le -2 ];then
echo ============================================================================
echo "             Train and decode neural network acoustic model               "
echo ============================================================================

[ -f path.sh ] && source path.sh

CUDA_VISIBLE_DEVICES=$gpu_ids python3 main.py
fi

