#!/bin/bash

#
# Copyright 2013 Bagher BabaAli,
#           2014 Brno University of Technology (Author: Karel Vesely)
#
# TIMIT, description of the database:
# http://perso.limsi.fr/lamel/TIMIT_NISTIR4930.pdf
#
# Hon and Lee paper on TIMIT, 1988, introduces mapping to 48 training phonemes, 
# then re-mapping to 39 phonemes for scoring:
# http://repository.cmu.edu/cgi/viewcontent.cgi?article=2768&context=compsci
#
echo "TIMIT Job"
. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

# Acoustic model parameters
numLeavesTri1=2500
numGaussTri1=15000
numLeavesMLLT=2500
numGaussMLLT=15000
numLeavesSAT=2500
numGaussSAT=15000
numGaussUBM=400
numLeavesSGMM=7000
numGaussSGMM=9000

feats_nj=30
train_nj=30
decode_nj=20

echo ============================================================================
echo "                Data & Lexicon & Language Preparation                     "
echo ============================================================================

# timit=/export/data/TIMIT/TIMIT/david/ma_ssp/2007/TIMIT
# local/timit_data_prep.sh $timit || exit 1

# local/timit_prepare_dict.sh

# utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
# data/local/dict "sil" data/local/lang_tmp data/lang

# local/timit_format_data.sh

echo ============================================================================
echo "         FBANK Feature Extration & CMVN for Training and Test set          "
echo ============================================================================

# fbankdir=fbank

# # 生成fbank特征
# for x in train dev test; do
    # steps/make_fbank.sh --cmd "$train_cmd" --nj $feats_nj --compress false data/$x exp/make_fbank/$x $fbankdir
    # steps/compute_cmvn_stats.sh data/$x exp/make_fbank/$x $fbankdir
# done

echo ============================================================================
echo "                     MonoPhone Training & Decoding                        "
echo ============================================================================

# steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train data/lang exp/mono

utils/mkgraph.sh --mono data/lang_test_bg exp/mono exp/mono/graph

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
exp/mono/graph data/dev exp/mono/decode_dev

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
exp/mono/graph data/test exp/mono/decode_test

echo ============================================================================
echo "           tri1 : Deltas + Delta-Deltas Training & Decoding               "
echo ============================================================================

steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
data/train data/lang exp/mono exp/mono_ali

steps/train_deltas.sh --cmd "$train_cmd" \
$numLeavesTri1 $numGaussTri1 data/train data/lang exp/mono_ali exp/tri1

utils/mkgraph.sh data/lang_test_bg exp/tri1 exp/tri1/graph

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
exp/tri1/graph data/dev exp/tri1/decode_dev

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
exp/tri1/graph data/test exp/tri1/decode_test

# echo ============================================================================
# echo "                 tri2 : LDA + MLLT Training & Decoding                    "
# echo ============================================================================



# steps/train_lda_mllt.sh --cmd "$train_cmd" \
# --splice-opts "--left-context=3 --right-context=3" \
# $numLeavesMLLT $numGaussMLLT data/train data/lang exp/tri1_ali exp/tri2

# utils/mkgraph.sh data/lang_test_bg exp/tri2 exp/tri2/graph

# steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# exp/tri2/graph data/dev exp/tri2/decode_dev

# steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# exp/tri2/graph data/test exp/tri2/decode_test

echo ============================================================================
echo "              tri3 : LDA + MLLT + SAT Training & Decoding                 "
echo ============================================================================

steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
 data/train data/lang exp/tri1 exp/tri1_ali
 
steps/train_sat.sh --cmd "$train_cmd" \
$numLeavesSAT $numGaussSAT data/train data/lang exp/tri1_ali exp/tri3

utils/mkgraph.sh data/lang_test_bg exp/tri3 exp/tri3/graph

steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
exp/tri3/graph data/dev exp/tri3/decode_dev

steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
exp/tri3/graph data/test exp/tri3/decode_test

  
echo ============================================================================
echo "                        tri3 ali Training & Decoding                         "
echo ============================================================================

steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
data/train data/lang exp/tri3 exp/tri3_ali
