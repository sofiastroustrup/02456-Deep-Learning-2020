#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:01:50 2020

@author: sofiastroustrup
"""
#import data_functions
import torch
from data_functions import read_fasta2, get_inputs_targets_from_sequences, \
    one_hot_encode_sequence, data_split, IUPAC_VOCAB, organize_fasta_header, get_train_test

#sequences=read_fasta("data/*.fasta")
sequences, description=read_fasta2("data/*.fasta")

inputs, targets = get_inputs_targets_from_sequences(sequences)
targets_idx=[torch.tensor([IUPAC_VOCAB[i] for i in sequence]) for sequence in targets]


inputs=[one_hot_encode_sequence(sequence, vocab=IUPAC_VOCAB) for sequence in inputs]

#targets=[one_hot_encode_sequence(sequence, vocab=IUPAC_VOCAB) for sequence in targets]

comb_tar_des=zip(targets_idx, description)

#train, val, test = data_split(inputs, targets_idx, train_frac=0.8)
#%%
#train, val, test = data_split(inputs, list(comb_tar_des), train_frac=0.8)

train, test=get_train_test(inputs, list(comb_tar_des), train_frac=0.9)
#%%
## prep train test such that we can get metadata
unzip_train1 = list(zip(*train[1]))
train_out= unzip_train1[0]
train=(train[0],train_out)

#unzip_val1 = list(zip(*val[1]))
#val_out= unzip_val1[0]
#val=(val[0],val_out)


unzip_test1 = list(zip(*test[1]))
test_out= unzip_test1[0]
test=(test[0],test_out)
test_des = unzip_test1[1]

test_info = organize_fasta_header(test_des)


