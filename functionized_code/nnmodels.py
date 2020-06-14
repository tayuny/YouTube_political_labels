import numpy as np
import pandas as pd
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

###################################################################################
#The following code is inspired by and modified from the PyTorch Tutorial of Ben Trevett, and assignment code of CAPP 30255, part of the modification will be marked with comments

# First Source:
# Topic: Tutorials on getting started with PyTorch and TorchText for sentiment analysis
# Source: https://github.com/bentrevett/pytorch-sentiment-analysis
# Author: Ben Trevett
# Date: 2019

# Second Source:
# Topic: Assignment 2 of CAPP 30255, The University of Chicago
# Author: Amitabh Chaudhary
# Date: 2020
####################################################################################

class SimpleRNN(nn.Module):
    '''
    Simple Recurrent Neural Network model without dropout setting
    '''
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.RNN = nn.RNN(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)  
        self.relu = nn.ReLU()
        
    def forward(self, text):
        
        #print("simpleRNN text:, ", text.size())
        #self.GRU.flatten_parameters()
        emb = self.embedding(text)
        ot1, hidden = self.RNN(emb)
        ot2 = self.relu(hidden.squeeze(0))
        output = self.linear(ot2)
        
        return output


class LSTM(nn.Module):
    '''
    Long Short Term Memory model with dropout setting
    '''
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, 
                 n_layers, bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.LSTM = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, text):
        
        # Simplify the original version of LSTM implementation
        #self.LSTM.flatten_parameters()
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.LSTM(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        final_out = hidden #self.relu(hidden)
            
        return self.linear(final_out)


class CNN(nn.Module):
    '''
    Convolutional Neural Network model with n filters
    '''
    def __init__(self, input_dim, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
                
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                
        embedded = self.embedding(torch.transpose(text, 0, 1))
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
            
        return self.fc(cat)


class WordEmbAvg_2linear(nn.Module):
    '''
    The 2-layer Linear Neural Network model with average embedding setting
    '''
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, pad_idx):
        
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)  
        self.relu = nn.ReLU()
                                 
    def forward(self, text):
        
        # Modify the original version of the CAPP 30255 assignment
        emb = self.embedding(text)
        emb = torch.mean(emb, dim=0).squeeze(1)
        ot1 = self.linear1(emb)
        ot2 = self.relu(ot1)
        output = self.linear2(ot2)
        
        return output


# The following code is inspired by and modified from the PyTorch Tutorial of Robert Guthrie, 
# part of the modification will be marked with comments.
#######################################################################################
# Topic: DEEP LEARNING WITH PYTORCH
# Author: Robert Guthrie
# Source: https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#sphx-glr-beginner-nlp-deep-learning-tutorial-py
# Date: 2017
########################################################################################
class BoWNN(nn.Module):
    '''
    The linear neural network model with bag of words setting
    '''
    def __init__(self, input_size, output_size):

        super(BoWNN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, bow_vec):
        
        return self.linear(bow_vec)


# The following code is inspired by and modified from the GRU Tutorial from Gabriel Loye, 
# part of the modification will be marked with comments.
#####################################################################################
# Topic: Gated Recurrent Unit (GRU) With PyTorch
# Author: Gabriel Loye
# Source: https://blog.floydhub.com/gru-with-pytorch/
# Date: 2020
######################################################################################

class GRU(nn.Module):
    '''
    Gated Recurrent Unit Model with dropout setting
    '''
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):

        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.GRU = nn.GRU(embedding_dim, 
                          hidden_dim, 
                          n_layers, 
                          bidirectional=bidirectional, 
                          dropout=dropout)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, text):

        # Modify the original version of GRU implementation
        #self.GRU.flatten_parameters()
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.GRU(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        final_out = hidden #self.relu(hidden)
            
        return self.linear(final_out)
