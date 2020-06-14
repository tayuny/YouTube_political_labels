import numpy as np
import pandas as pd
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext import data
import torch.utils.data as d
import tokenization_dim_reduction as tdr


def binary_accuracy(preds, y):
    """
    Return accuracy per batch
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def binary_precision(preds, y):
    '''
    Return precision per batch
    '''
    rounded_preds = torch.round(torch.sigmoid(preds))
    prec_correct = ((rounded_preds == y) & (rounded_preds == 1)).float()
    prec_total = (rounded_preds == 1).float()
    precision = prec_correct.sum() / prec_total.sum()
    return precision


def binary_recall(preds, y):
    '''
    Return recall per batch
    '''
    rounded_preds = torch.round(torch.sigmoid(preds))
    rec_correct = ((rounded_preds == y) & (rounded_preds == 1)).float()
    rec_total = (y == 1).float()
    recall = rec_correct.sum() / rec_total.sum()
    return recall


###################################################################################
# The code for Training_module object is inspired by and modified from the PyTorch 
# Tutorial of Ben Trevett, and assignment code of CAPP 30255, part of the modification 
# will be marked with comments

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
class Training_module( ):

    def __init__(self, model, device):
        
        self.model = model
        self.loss_fn = (nn.BCEWithLogitsLoss()).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
    
    def train_epoch(self, iterator):
        '''
        Train the model for one epoch. For this repeat the following, 
        going through all training examples.
        1. Get the next batch of inputs from the iterator.
        2. Determine the predictions using a forward pass.
        3. Compute the loss.
        4. Compute gradients using a backward pass.
        5. Execute one step of the optimizer to update the model paramters.
        '''
        epoch_loss = 0
        epoch_acc = 0
    
        for batch in iterator:

            self.optimizer.zero_grad()
            
            predictions = self.model(batch.text).squeeze(1)
            loss = self.loss_fn(predictions, batch.label)
            accuracy = binary_accuracy(predictions, batch.label)
        
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += accuracy.item()
        
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    def train_model(self, train_iterator, dev_iterator, models_perf, key, epoch_num=5):
        """
        Train the model for multiple epochs, and after each evaluate on the
        development set.  Return the best performing model and updated performance
        dictionary for each epoch
        """  
        dev_accs = [0.]
        for epoch in range(epoch_num):
            self.train_epoch(train_iterator)
            dev_acc = self.evaluate(dev_iterator)

            print(f"Epoch {epoch}: Dev Accuracy: {dev_acc[1]} Dev Loss:{dev_acc[0]}")
            models_perf[key]["valid_epoch_acc " + str(epoch)] = dev_acc[1]

            if dev_acc[1] > max(dev_accs):
                best_model = copy.deepcopy(self)
            dev_accs.append(dev_acc[1])

        return best_model.model, models_perf
                
    def evaluate(self, iterator):
        '''
        Evaluate the performance of the model on the given examples.
        '''
        epoch_loss = 0
        epoch_acc = 0
        epoch_prec = 0
        epoch_rec = 0
    
        with torch.no_grad():
    
            for batch in iterator:

                predictions = self.model(batch.text).squeeze(1)
                loss = self.loss_fn(predictions, batch.label)
                acc = binary_accuracy(predictions, batch.label)
                precision = binary_precision(predictions, batch.label)
                recall = binary_recall(predictions, batch.label)
        
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                epoch_prec += precision.item()
                epoch_rec += recall.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator), \
               epoch_prec / len(iterator), epoch_rec / len(iterator)


def model_selection(model_dict, model_txt="avg_embedding"):
    '''
    Helper function for model selection
    '''
    return model_dict[model_txt]


# Comparing the performance of different models, the original 
# version of the CAPP 30255 assignment
def compare_models(model_dict, device, train_iterator, valid_iterator, test_iterator, epoch_num=5):
    '''
    The function presents and compare the performances of
    different neural network models and store the best
    models of each model type in the output dictionary
    
    Inputs: 
        model_dict: dictionary of model types used in training
        device: the device object assigned for the operation
        train_iterator, valid_iterator, test_iterator: iterators with batches
    Return: dictionary of best models of each model type
    '''
    models_perf = {}
    best_models = {}
    for key, value in model_dict.items():
        print("currently training the model: ", key)
        models_perf[key] = {}

        model = model_selection(model_dict, key)
        model = model.to(device)
        tm = Training_module(model, device)
        best_model, models_perf = tm.train_model(train_iterator, valid_iterator, models_perf, key, epoch_num)
        best_models[key] = best_model
        
        tm.model = best_model
        test_loss, test_acc, test_prec, test_rec = tm.evaluate(test_iterator)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
        print(f'Test Prec: {test_prec*100:.3f}% | Test Rec: {test_rec*100:.3f}%')

        models_perf[key]["best_test_acc"] = test_acc
        models_perf[key]["best_test_precision"] = test_prec
        models_perf[key]["best_test_recall"] = test_rec
        
    return best_models, models_perf


# Searching for phrases with highest norm values, modify the original 
# version of the CAPP 30255 assignment
def get_effective_norms(best_models, TEXT, selected_mkey="avg_embedding"):
    '''
    The model presents 10 most effective and 10 less effective
    phrases used in the classification
    Inputs: 
        best_models: dictionary of best model of each model type
        TEXT: TEXT object used in vocabulary building
        selected_mkey: selected model type
    '''
    best_model = best_models[selected_mkey]
    strong_words = []
    weak_words = []
    emb_weight = best_model.embedding.weight.data
    top_indices = torch.norm(emb_weight, p=2, dim=1).detach().topk(10).indices
    bottom_indices = torch.norm(emb_weight, p=2, dim=1).detach().topk(10, largest=False).indices

    for idx in top_indices:
        strong_words.append(TEXT.vocab.itos[idx])
    
    for idx in bottom_indices:
        weak_words.append(TEXT.vocab.itos[idx])
    
    print("most effective words: ", strong_words)
    print("less effective words: ", weak_words)