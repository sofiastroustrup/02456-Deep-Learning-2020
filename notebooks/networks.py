#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:41:17 2020

@author: sofiastroustrup
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence




class GruNet_packed_seq(nn.Module):
    def __init__(self, vocab_size, n_layers, hidden_size, batch_size):
        super(GruNet_packed_seq, self).__init__()
        self.vocab_size=vocab_size
        self.n_layers=n_layers
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        
        # Recurrent layer
        self.gru = nn.GRU(input_size=self.vocab_size,
                         hidden_size=self.hidden_size,
                         num_layers=self.n_layers,
                         bidirectional=False,dropout=0.5)
        


        # Output layer
        self.l_out = nn.Linear(in_features=self.hidden_size,
                            out_features=self.vocab_size,
                            bias=False)
        
    def forward(self, x):
        """x: packed sequence"""
        
        # RNN returns output and last hidden state
        x, h = self.gru(x)
        
        # tilføj relu
        
        if type(x)==torch.nn.utils.rnn.PackedSequence:
            x, seq_len= torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        x=F.relu(x)

        #x = x.contiguous()
        #X = X.view(-1, X.shape[2])
        # Flatten output for feed-forward layer
        x = x.view(-1, self.gru.hidden_size)
        #print(x.shape)
        # Output layer
        x = self.l_out(x)
        #print(x.shape)
        #x = x.view(self.batch_size, max(seq_len), self.vocab_size)
        x = x.view(-1, self.vocab_size)
        x = F.log_softmax(x, dim=1)

        # print(x.shape)
        return x, h





# define GRU
class GruNet(nn.Module):
    def __init__(self, vocab_size, n_layers, hidden_size):
        super(GruNet, self).__init__()
        self.vocab_size=vocab_size
        self.n_layers=n_layers
        self.hidden_size = hidden_size
        
        # Recurrent layer
        self.gru = nn.GRU(input_size=self.vocab_size,
                         hidden_size=self.hidden_size,
                         num_layers=self.n_layers,
                         bidirectional=False, dropout=0.5)
        
        # Output layer
        self.l_out = nn.Linear(in_features=self.hidden_size,
                            out_features=self.vocab_size,
                            bias=False)
        
    def forward(self, x):
        # RNN returns output and last hidden state
        x, h = self.gru(x)
        
        # Flatten output for feed-forward layer
        x = x.view(-1, self.gru.hidden_size)
        
        # Output layer
        x = self.l_out(x)
        
        return x





class Gru(nn.Module):
    def __init__(self, vocab, n_layers, hidden_size, batch_size, embedding_dim=10):
        super(Gru, self).__init__()
        self.vocab_size=len(vocab.values())
        self.vocab = vocab
        self.n_layers=n_layers
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.dropout = nn.Dropout(p=0.5)
        self.embedding_dim = embedding_dim

        # Recurrent layer

        padding_idx = self.vocab['<pad>']
        self.word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx
        )


        self.linear1 = nn.Linear(in_features = self.embedding_dim, out_features=30)
        self.linear2 = nn.Linear(in_features = 30, out_features=40)

        self.gru = nn.GRU(input_size=40,
                         hidden_size=self.hidden_size,
                         num_layers=self.n_layers,
                         bidirectional=False,dropout=0.5)
        


        # Output layer
        self.l_out = nn.Linear(in_features=self.hidden_size,
                            out_features=self.vocab_size,
                            bias=False)
        
    def forward(self, x):
        """x: packed sequence"""
        ps=None
        if type(x)==torch.nn.utils.rnn.PackedSequence:
            ps=True
            x, seq_len= torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
        
        x = self.word_embedding(x)
        x=self.dropout(F.relu(self.linear1(x)))
        x=self.dropout(F.relu(self.linear2(x)))

        if ps:
            x=torch.nn.utils.rnn.pack_padded_sequence(x, seq_len, enforce_sorted=False)
        
        if not ps:
            x=x.reshape(-1,1,40)

        # RNN returns output and last hidden state
        x, h = self.gru(x)
        
        # tilføj relu
        
        if type(x)==torch.nn.utils.rnn.PackedSequence:
            x, seq_len= torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        x=F.relu(x)

        # Flatten output for feed-forward layer
        x = x.view(-1, self.gru.hidden_size)

        # Output layer
        x = self.l_out(x)

        x = x.view(-1, self.vocab_size)
        x = F.log_softmax(x, dim=1)

        # print(x.shape)
        return x, h


import torch.nn.functional as F

class lstm_net(nn.Module):
    def __init__(self, vocab, n_layers, hidden_size, batch_size, embedding_dim=10):
        super(lstm_net, self).__init__()
        self.vocab_size=len(vocab.values())
        self.vocab = vocab
        self.n_layers=n_layers
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.dropout = nn.Dropout(p=0.5)
        self.embedding_dim = embedding_dim

        padding_idx = self.vocab['<pad>']
        self.word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx)

        self.lstm=nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.n_layers, dropout=0.5)

        self.l_out = nn.Linear(in_features=self.hidden_size,
                            out_features=self.vocab_size,
                            bias=False)
       
        
    def forward(self, x):
        """x: packed sequence"""
        
        ps=None
        if type(x)==torch.nn.utils.rnn.PackedSequence:
            ps=True
            x, seq_len= torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
        
        x = self.word_embedding(x)

        if ps:
            x=torch.nn.utils.rnn.pack_padded_sequence(x, seq_len, enforce_sorted=False)
        
        if not ps:
            x=x.reshape(-1,1,self.embedding_dim)


        # RNN returns output and last hidden state
        cs, h = self.lstm(x)
        x=cs
        # tilføj relu
        
        if type(x)==torch.nn.utils.rnn.PackedSequence:
            x, seq_len= torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        x=F.relu(x)

        #x = x.contiguous()
        #X = X.view(-1, X.shape[2])
        # Flatten output for feed-forward layer
        x = x.view(-1, self.lstm.hidden_size)
        #print(x.shape)
        # Output layer
        x = self.l_out(x)
        #print(x.shape)
        #x = x.view(self.batch_size, max(seq_len), self.vocab_size)
        x = x.view(-1, self.vocab_size)
        x = F.log_softmax(x, dim=1)

        # print(x.shape)
        return x, h
