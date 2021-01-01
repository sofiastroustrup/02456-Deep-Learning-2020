#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:55:36 2020

@author: sofiastroustrup
"""
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch



class Dataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # Return the size of the dataset
        return len(self.targets)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        X = self.inputs[index]
        y = self.targets[index]

        return X, y

def my_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    #[print(item[0].shape) for item in batch]
    #[print(len(item[1])) for item in batch]
    data = [item[0] for item in batch]
    data1=data
    seq_len = [item[0].shape[0] for item in batch]
    #seq_len_targets = [len(item[0][1]) for item in batch]
    #print(len(batch))
    #print(seq_len)
    p_data=pad_sequence(data)
    #print(p_data.shape)
    #print(p_data.shape)
    #data = pack_sequence(data, enforce_sorted=False)
    data=pack_padded_sequence(p_data, seq_len, enforce_sorted=False)
    #targets = [torch.tensor(item[1]) for item in batch]
    #targets = [item[1].clone().detach() for item in batch]
    targets = [item[1] for item in batch]
    
    targets = pad_sequence(targets)
    return [data, targets, data1]



def my_collate2(batch):
    # batch contains a list of tuples of structure (sequence, target)
    #[print(item[0].shape) for item in batch]
    #[print(len(item[1])) for item in batch]
    data = [item[0] for item in batch]
    data1=data
    seq_len = [item[0].shape[0] for item in batch]
    #seq_len_targets = [len(item[0][1]) for item in batch]
    #print(len(batch))
    #print(seq_len)
    p_data=pad_sequence(data)
    #print(p_data.shape)
    #print(p_data.shape)
    #data = pack_sequence(data, enforce_sorted=False)
    data=pack_padded_sequence(p_data, seq_len, enforce_sorted=False)
    #targets = [torch.tensor(item[1]) for item in batch]
    #targets = [item[1].clone().detach() for item in batch]
    targets = [item[1] for item in batch]
    
    targets = pad_sequence(targets)
    return [data, targets, data1]




