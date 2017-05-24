#!/bin/bash

# Begin configuration section.  
nj=4
cmd=run.pl
apply_deltas=false
cmvn_opts=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euo pipefail

data=$1
srcdata=$2
logdir=$3
feadir=$4

sdata=$srcdata/split$nj;

# 决定是否重新拆分数据
mkdir -p $data $logdir $feadir
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

# Check files exist,
for f in $sdata/1/feats.scp $sdata/1/cmvn.scp; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

if $apply_deltas; then
    echo "apply delta tran"
    feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas --delta-order=2 ark:- ark:- |"
else
    echo "without delta tran"
    feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
fi
# Prepare the output dir,
# 清除data目录下残留的feats和cmvn文件
utils/copy_data_dir.sh $srcdata $data; rm $data/{feats,cmvn}.scp 2>/dev/null
# Make $feadir an absolute pathname,
[ '/' != ${feadir:0:1} ] && feadir=$PWD/$feadir

# Store the output-features,
name=`basename $data`
$cmd JOB=1:$nj $logdir/make_fbank_feats.JOB.log \
  copy-feats "$feats" \
  ark,scp:$feadir/feats_fbank_$name.JOB.ark,$feadir/feats_fbank_$name.JOB.scp || exit 1;
   
# Merge the scp,
for n in $(seq 1 $nj); do
  cat $feadir/feats_fbank_$name.$n.scp 
done > $srcdata/feats.scp

exit 0;
