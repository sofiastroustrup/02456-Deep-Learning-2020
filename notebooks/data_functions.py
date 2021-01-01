#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:02:53 2020
file that defines functions used for handling data
"""
from io import open
import glob
import os
from Bio import SeqIO
from collections import OrderedDict
import sklearn.model_selection
import torch
import pandas as pd 

def read_fasta(path):
    """
    input:
    output:
    """
    filename=glob.glob(path)
    # parse the fasta files 
    sequences=[]
    for j in range(len(filename)):
        records = list(SeqIO.parse(filename[j], "fasta"))
        n_seq=len(records)
        for i in range(n_seq):
            if len(records[i].seq)<500 and len(records[i].seq)>100:
                cur_seq=[str(s) for s in list(records[i].seq)]
                cur_seq.append('<unk>')# append end token 
                sequences.append(cur_seq)
    return(sequences)

def read_fasta2(path):
    """
    input:
    output:
    """
    filename=glob.glob(path)
    # parse the fasta files 
    sequences=[]
    info=[]
    for j in range(len(filename)):
        records = list(SeqIO.parse(filename[j], "fasta"))
        n_seq=len(records)
        for i in range(n_seq):
            if len(records[i].seq)<500 and len(records[i].seq)>20:
                cur_seq=[str(s) for s in list(records[i].seq)]
                cur_seq.append('<unk>')# append end token 
                sequences.append(cur_seq)
                info.append(records[i].description)
    return(sequences,info)





def get_inputs_targets_from_sequences(sequences):
    """ 
    Run this function on a list of sequences to get the (X,y) tuples for each sequence
    Input:
    Output:
    """
    # Define empty lists
    inputs, targets = [], []
    
    # Append inputs and targets s.t. both lists contain L-1 words of a sentence of length L
    # but targets are shifted right by one so that we can predict the next word
    for sequence in sequences:
        inputs.append(sequence[:-1])
        targets.append(sequence[1:])
        
    return inputs, targets


def data_split(examples, labels, train_frac, random_state=None):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    X_train, X_tmp, Y_train, Y_tmp = sklearn.model_selection.train_test_split(
                                        examples, labels, train_size=train_frac, random_state=random_state)

    X_val, X_test, Y_val, Y_test   = sklearn.model_selection.train_test_split(
                                        X_tmp, Y_tmp, train_size=0.5, random_state=random_state)
    
    training_set = (X_train, Y_train)
    validation_set = (X_val, Y_val)
    test_set = (X_test, Y_test)

    return training_set, validation_set, test_set


def get_train_test(examples, labels, train_frac, random_state=None):
    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
                                        examples, labels, train_size=train_frac, random_state=random_state)

    training_set = (X_train, Y_train)
    test_set = (X_test, Y_test)
    
    return(training_set, test_set)



def one_hot_encode_sequence(sequence, vocab):
    """ 
    Input:
    Output:
    
    """
    tensor = torch.zeros(len(sequence), len(vocab))
    # Encode each word in the sentence
    for li, word in enumerate(sequence):
        if vocab.get(word) is not None:
            #tensor[li][0][self.word_to_idx[word]]=1
            tensor[li][vocab[word]]=1
        else: 
            return("letter {} is not in dictionary".format(word))
    
    return(tensor)


def embed_sequence(sequence):
    """
    input:
    output:
    """
    #embed sequence, same input/output format as one_hot (but different vocab_size)


import re
#pattern="OS="
#p="(?<=OS=).*[1,9]"
#re.search(r'OS=(.*?)OX=', s)
def organize_fasta_header(headers):
    """Input: list of strings that are fasta headers """
    out_list = []
    for string in headers:
        #p_id = r'|(.*?)|'
        if re.search(r'tr\|(.*?)\|', string) is not None:
          uniq_id = re.search(r'tr\|(.*?)\|', string).group(1)
        else:
          uniq_id = "NA"
        org_name = re.search(r'OS=(.*?)OX=', string).group(1)
        org_id = re.search(r'OX=(.*?)\s', string).group(1)
        if "GN" in string and re.search(r'GN=(.*?)\s', string) is not None:
            gene_name = re.search(r'GN=(.*?)\s', string).group(1)
        else:
            gene_name= "NA"
        out_list.append([uniq_id, org_name, org_id, gene_name])
    
    metadata =pd.DataFrame(out_list, columns=["uniq_id", "organism", "organism_id", "gene_name"])
    return(metadata)
    
    
    
    
    





IUPAC_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("<unk>", 4),
    ("A", 5),
    ("B", 6),
    ("C", 7),
    ("D", 8),
    ("E", 9),
    ("F", 10),
    ("G", 11),
    ("H", 12),
    ("I", 13),
    ("K", 14),
    ("L", 15),
    ("M", 16),
    ("N", 17),
    ("O", 18),
    ("P", 19),
    ("Q", 20),
    ("R", 21),
    ("S", 22),
    ("T", 23),
    ("U", 24),
    ("V", 25),
    ("W", 26),
    ("X", 27),
    ("Y", 28),
    ("Z", 29)])












