'''@file main.py
run this file to go through the neural net training procedure, look at the config files in the config directory to modify the settings'''
# -*- coding: utf-8 -*-

import sys
import os
import shutil

os.system('. ./path.sh')
sys.path.append('local/kaldi')
sys.path.append('local/processing')
sys.path.append('local/neuralNetworks')
sys.path.append('local/neuralNetworks/classifiers')
from six.moves import configparser
import numpy as np
import tensorflow as tf
from time import clock

import ark, prepare_data, feature_reader, batchdispenser, target_coder
import nnet

#here you can set which steps should be executed. If a step has been executed in the past the result have been saved and the step does not have to be executed again (if nothing has changed)
TRAIN_NNET = True
TEST_NNET = True
DEV_NNET = True

#read config file
config = configparser.ConfigParser()
config.read('conf/cnn.cfg')
cnn_conf = dict(config.items('nnet'))
current_dir = os.getcwd()
# the name of gmm
gmm_name = cnn_conf['gmm_name']
# the name of nnet
nnet_name = cnn_conf['name']
# 后台读取batch的线程数
batch_reader_nj = 8
# the num job in gmm ali
num_ali_jobs = int(config.get('directories','num_ali_jobs'))
# the context
context_left = int(config.get('nnet','context_left'))
context_right = int(config.get('nnet','context_right'))
total_context = context_left + context_right + 1
# train 特征目录
train_features_dir = config.get('directories','train_features')
test_features_dir = config.get('directories','test_features')
dev_features_dir = config.get('directories','dev_features')
# exp dir to get data
expdir = config.get('directories','expdir')
# exp dir to store data
store_expdir = config.get('directories','store_expdir')
# the ali dir
alidir = config.get('directories','alidir')

if not os.path.isdir(store_expdir):
    os.mkdir(store_expdir)
if not os.path.isdir(store_expdir + '/' + nnet_name):
    os.mkdir(store_expdir + '/' + nnet_name)

# 网络输入维度
reader = ark.ArkReader(train_features_dir + '/feats.scp')
_, features, _ = reader.read_next_utt()     # 这里是没有经过拼接的
input_dim = features.shape[1] * total_context
print("the input dim is:" + str(input_dim))

# 网络输出维度
numpdfs = open(expdir + '/' + gmm_name + '/graph/num_pdfs')
num_labels = numpdfs.read()
num_labels = int(num_labels[0:len(num_labels)-1])
numpdfs.close()
print("the output labels is:" + str(num_labels))

# get the maxlength of all utt
max_input_length = 0
total_frames = 0
with open(train_features_dir + "/utt2num_frames", 'r') as f:
    line = f.readline()
    while line:
        x = line.split(' ')[1]
        total_frames += int(x)
        if int(x) > max_input_length:
            max_input_length = int(x)
        line = f.readline()
# 将maxlength写入文件    
with open(train_features_dir + "/maxlength", 'w') as f:
    f.write("%s"%max_input_length)
    print("the utt's maxlength is: " + str(max_input_length))   
with open(train_features_dir + "/total_frames", 'w') as f:
    f.write("%s"%total_frames)
    print("the total frame in training set is: " + str(total_frames))
    
# pad the maxlength, so we can use easily divide the whole seq to sub-seq
if dict(config.items('nnet'))['lstm_type'] == 3:
    print("Use the lstm type 3")
    time_steps = int(dict(config.items('lstm3'))['time_steps'])
    print("And we will pad the max-length of all utt to be divisible by " + str(time_steps))
    max_input_length = max_input_length - max_input_length%time_steps + time_steps
    

nnet = nnet.Nnet(config, input_dim, num_labels, max_input_length)

if TRAIN_NNET:
    # shuffle the examples on disk
    prepare_data.shuffle_examples(train_features_dir)
      
    print('------- get alignments ----------')
    alifiles = [ expdir + '/' + alidir + '/ali.' + str(i+1) + '.gz' for i in range(num_ali_jobs)]
    alifilebinary = store_expdir + '/' + nnet_name + '/ali.binary.gz'
    alifile = store_expdir + '/' + nnet_name + '/ali.text.gz'
    if not os.path.isfile(alifile):
        tmp = open(alifile, 'a')
        tmp.close()
        tmp = open(alifilebinary, 'a')
        tmp.close()
    os.system('cat %s > %s' % (' '.join(alifiles), alifilebinary))
    ## debug
    os.system('copy-int-vector ark:"gunzip -c %s |" ark:- | ali-to-pdf %s/final.mdl ark:- ark,t:- | gzip -c > %s'%(alifilebinary,expdir + '/' + alidir,alifile))
    #os.system('copy-int-vector ark:"gunzip -c %s |" ark:- | ali-to-pdf %s/final.alimdl ark:- ark,t:- | gzip -c > %s'%(alifilebinary,expdir + '/' + alidir,alifile))
    


    # Here we directly use the feats in fmllr
    # So we ignore the cmvn process
    featreader = feature_reader.FeatureReader(train_features_dir + '/feats_shuffled.scp', 
        train_features_dir + '/utt2spk', context_left, context_right, max_input_length)
    
    # create a target coder
    coder = target_coder.AlignmentCoder(lambda x, y: x, num_labels)
    
    # lda在哪里做？
    dispenser = batchdispenser.AlignmentBatchDispenser(featreader, coder, 
        int(cnn_conf['batch_size']), input_dim, alifile)

    #train the neural net
    print('------- training neural net ----------')
    
    # use tf-kaldi to process the nnet
    start = clock()
    nnet.train(dispenser)
    end = clock()
    print('the nnet training time is : ' + str((end-start)/60) + '/min')
    
if DEV_NNET:
    start = clock()
    #use the neural net to calculate posteriors for the testing set
    print('------- Dev: computing state pseudo-likelihoods ----------')
    savedir = store_expdir + '/' + config.get('nnet', 'name')
    decodedir = savedir + '/decode_dev'
    if not os.path.isdir(decodedir):
        os.mkdir(decodedir)

    # get maxlength
    max_input_length = 0
    total_frames = 0
    with open(dev_features_dir + "/utt2num_frames", 'r') as f:
        line = f.readline()
        while line:
            x = line.split(' ')[1]
            total_frames += int(x)
            if int(x) > max_input_length:
                max_input_length = int(x)
            line = f.readline()
    # 将maxlength写入文件    
    with open(dev_features_dir + "/maxlength", 'w') as f:
        f.write("%s"%max_input_length)
        print("the utt's maxlength is: " + str(max_input_length))

    #create a feature reader
    with open(dev_features_dir + '/maxlength', 'r') as fid:
        max_length = int(fid.read())
    featreader = feature_reader.FeatureReader(dev_features_dir + '/feats.scp', dev_features_dir + '/utt2spk', 
        context_left, context_right, max_length)

    #create an ark writer for the likelihoods
    if os.path.isfile(decodedir + '/likelihoods.ark'):
        os.remove(decodedir + '/likelihoods.ark')
    writer = ark.ArkWriter(decodedir + '/feats.scp', decodedir + '/likelihoods.ark')

    #decode with te neural net
    nnet.decode(featreader, writer)

    print('------- decoding dev sets ----------')
    #copy the gmm model and some files to speaker mapping to the decoding dir
    os.system('cp %s %s' %(config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/final.mdl', decodedir))
    os.system('cp -r %s %s' %(config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/graph', decodedir))
    os.system('cp %s %s' %(config.get('directories', 'dev_features') + '/utt2spk', decodedir))
    os.system('cp %s %s' %(config.get('directories', 'dev_features') + '/text', decodedir))
    os.system('cp %s %s' %(config.get('directories', 'dev_features') + '/stm', decodedir))
    os.system('cp %s %s' %(config.get('directories', 'dev_features') + '/glm', decodedir))

    #change directory to kaldi egs
    os.chdir(config.get('directories', 'prjdir'))

    #decode using kaldi
    if not os.path.isfile(decodedir + "/decode.log"):
        os.system('touch %s/decode.log' % (decodedir))
    os.system('%s/local/kaldi/decode.sh --nj %s %s/graph %s %s/kaldi_decode | tee %s/decode.log || exit 1;' % 
        (current_dir, config.get('directories', 'decode_num_jobs'), 
            decodedir, decodedir, decodedir, decodedir))
    
    end = clock()
    print('the nnet decode time in dev is : ' + str((end-start)/60) + '/min')
    
if TEST_NNET:
    start = clock()
    #use the neural net to calculate posteriors for the testing set
    print('------- Test: computing state pseudo-likelihoods ----------')
    savedir = store_expdir + '/' + config.get('nnet', 'name')
    decodedir = savedir + '/decode_test'
    if not os.path.isdir(decodedir):
        os.mkdir(decodedir)

    # get maxlength
    max_input_length = 0
    total_frames = 0
    with open(test_features_dir + "/utt2num_frames", 'r') as f:
        line = f.readline()
        while line:
            x = line.split(' ')[1]
            total_frames += int(x)
            if int(x) > max_input_length:
                max_input_length = int(x)
            line = f.readline()
    # 将maxlength写入文件    
    with open(test_features_dir + "/maxlength", 'w') as f:
        f.write("%s"%max_input_length)
        print("the utt's maxlength is: " + str(max_input_length))

    #create a feature reader
    with open(test_features_dir + '/maxlength', 'r') as fid:
        max_length = int(fid.read())
    featreader = feature_reader.FeatureReader(test_features_dir + '/feats.scp', test_features_dir + '/utt2spk', 
        context_left, context_right, max_length)

    #create an ark writer for the likelihoods
    if os.path.isfile(decodedir + '/likelihoods.ark'):
        os.remove(decodedir + '/likelihoods.ark')
    writer = ark.ArkWriter(decodedir + '/feats.scp', decodedir + '/likelihoods.ark')

    print('get the likelihoods')
    #decode with te neural net
    nnet.decode(featreader, writer)

    print('------- decoding testing sets ----------')
    #copy the gmm model and some files to speaker mapping to the decoding dir
    os.system('cp %s %s' %(config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/final.mdl', decodedir))
    os.system('cp -r %s %s' %(config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/graph', decodedir))
    os.system('cp %s %s' %(config.get('directories', 'test_features') + '/utt2spk', decodedir))
    os.system('cp %s %s' %(config.get('directories', 'test_features') + '/text', decodedir))
    os.system('cp %s %s' %(config.get('directories', 'test_features') + '/stm', decodedir))
    os.system('cp %s %s' %(config.get('directories', 'test_features') + '/glm', decodedir))

    #change directory to kaldi egs
    os.chdir(config.get('directories', 'prjdir'))

    #decode using kaldi
    if not os.path.isfile(decodedir + "/decode.log"):
        os.system('touch %s/decode.log' % (decodedir))
    os.system('%s/local/kaldi/decode.sh --nj %s %s/graph %s %s/kaldi_decode | tee %s/decode.log || exit 1;' % 
        (current_dir, config.get('directories', 'decode_num_jobs'), 
            decodedir, decodedir, decodedir, decodedir))
            
    end = clock()
    print('the nnet decode time in test is : ' + str((end-start)/60) + '/min')

#get results
os.system('grep Sum tf-exp/cnn/decode_dev/*decode/score_*/*.sys 2>/dev/null | utils/best_wer.sh')
os.system('grep Sum tf-exp/cnn/decode_test/*decode/score_*/*.sys 2>/dev/null | utils/best_wer.sh')