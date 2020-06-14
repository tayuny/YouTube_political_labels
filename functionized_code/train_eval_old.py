from typing import Tuple, List, Dict, Set, Any, Type
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

    # Identify the device on which to put everything.
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __init__(self, model, device=DEVICE, silent=False):
        
        self.model = model.to(device=device)
        self.loss_fn = (nn.BCEWithLogitsLoss()).to(device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.silent = silent
    

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
        # Set the model to training mode.
        self.model.train()
        # Consider each batch of examples in the training data.
        for batch in iterator:
            # Clear the gradients from any previous optimization.
            self.optimizer.zero_grad()
            # Forward pass: get predictions from the model.
            predictions = self.model(batch.text).squeeze(1)
            loss = self.loss_fn(predictions, batch.label)
            # Backward pass: compute gradients and update model parameters.
            loss.backward()
            self.optimizer.step()
        # Return the trained model.
        return self.model
    

    def evaluate(self, iterator):
        '''
        Evaluate the performance of the model on the given examples.
        '''
        # Initialize aggregators for each metric.
        loss, accuracy, precision, recall = 0, 0, 0, 0
        # Set the model to evaluation mode.
        self.model.eval()
        # Disable gradient calculation.
        with torch.no_grad():
            # Consider each batch of examples in the validation data.
            for batch in iterator:
                # Get predictions from the model.
                predictions = self.model(batch.text).squeeze(1)
                # Increment each metric.
                loss += self.loss_fn(predictions, batch.label).item()
                accuracy += binary_accuracy(predictions, batch.label).item()
                precision += binary_precision(predictions, batch.label).item()
                recall += binary_recall(predictions, batch.label).item()
        # Collect the average of each metric. 
        n = len(iterator)
        metrics = {
            'loss': loss / n,
            'accuracy': accuracy / n,
            'precision': precision / n,
            'recall': recall / n
        }
        # Return the metrics.
        return metrics
    

    def train_model(self, itr_train, itr_valid, decision_metric='accuracy', epochs=5):
        '''
        Train and evaluate the model on the training and validation datasets for
        the number of epochs indicated. Choose the best model from the decision
        metric indicated, e.g. loss, accuracy, precision, recall.
        
        Return the metrics collected for each epoch and the best model (dict).
        '''
        # Initialize containers for performance metrics.
        metrics = {}
        decision_metrics = []
        best_model = None
        # Train the model for each epoch.
        for epoch in range(epochs):
            # Train the model with the training data.
            model = self.train_epoch(itr_train)
            # Evaluate the model with the validation data.
            metrics[epoch] = self.evaluate(itr_valid)
            # Save this model if has the best decision metric seen so far.
            decision_metrics.append(metrics[epoch][decision_metric])
            if decision_metric == 'loss':
                if decision_metrics[-1] <= min(decision_metrics):
                    best_model = copy.copy(model.state_dict())
            else:
                if decision_metrics[-1] >= max(decision_metrics):
                    best_model = copy.copy(model.state_dict())
            # Report performance with this epoch.
            if not self.silent:
                print_metrics('Epoch: %d' % epoch, metrics[epoch])
        # Pack the best model in with the metrics.
        metrics['best'] = best_model
        # Return the performance metrics.
        return metrics


# Comparing the performance of different models, the original 
# version of the CAPP 30255 assignment
def compare_models(model_dict, device, train_iterator, valid_iterator, test_iterator):
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
    best_models = {}
    models_perf = {}
    for name, model in model_dict.items():
        # Report the model on deck.
        print("Currently training the model: ", name)
        # Train this model on the training and validation sets.
        tm = Training_module(model, device)
        metrics = tm.train_model(train_iterator, valid_iterator)
        # Load and evaluate the best model on the testing set.
        tm.model.load_state_dict(metrics['best'])
        metrics['test'] = tm.evaluate(test_iterator)
        print_metrics('Testing', metrics['test'])
        # Update the collections of models and metrics.
        best_models[name] = copy.deepcopy(tm.model)
        models_perf[name] = metrics
    # Return the best model and the model metrics.
    return best_models, models_perf


def print_metrics(prefix: str, m: dict) -> None:
    print(
        '%s | Loss: %6.4f | Acc: %6.4f | Prec: %6.4f | Rec: %6.4f'
        % (prefix, m['loss'], m['accuracy'], m['precision'], m['recall'])
    )


# Searching for phrases with highest norm values, modify the original 
# version of the CAPP 30255 assignment
def get_effective_norms(model, idx_to_word, k):
    '''
    Report the top k most and least effective words in classification.

    model (torch.nn.Module): model from which to get word embeddings.
    idx_to_word (dict): mapping of indices to words in the vocabulary.
    k (int): number of words to report.

    '''
    # Calculate the norm of the word vectors in the embeddings.
    word_norms = torch.norm(model.embedding.weight, p=2, dim=1)
    # Get the vocabulary indices of the top-k and bottom-k words.
    top_indices = word_norms.topk(k).indices
    bottom_indices = word_norms.topk(k, largest=False).indices
    # Map these indices to words in the vocabulary.
    strong_words = [idx_to_word[idx.item()] for idx in top_indices]
    weak_words = [idx_to_word[idx.item()] for idx in bottom_indices]
    print("More effective words: ", strong_words)
    print("Less effective words: ", weak_words)
