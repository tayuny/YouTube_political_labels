import numpy as np
import pandas as pd
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tokenization_dim_reduction as tdr
import ngrams as ng

label_to_ix = {0.0: 0, 1.0: 1}

def ngram_creater(n, sentence, whole_grams):
    '''
    The function creates n-grams dictionary with its count
    Input:
        n: n for n-grams
        sentence: single sentence passed in
        whole_grams: dictionary with accumulated count
    Returns:
        grams: n-grams with its counts for the sentence
        updated whole_grams
    '''
    grams = {}
    word_lst = ng.clean_punctuation(sentence).split()
    for i in range(len(word_lst) - n + 1):
        gram = " ".join(word_lst[i:i + n])
        if not grams.get(gram):
            grams[gram] = 0
            whole_grams[gram] = 0
        grams[gram] += 1
        whole_grams[gram] += 1
    
    return grams, whole_grams


def word_ngrams(n, txt_arr):
    '''
    The function creates the ngrams with its corresponding
    counts for the whole dataset
    Inputs:
        n: n for n-grams
        txt_arr: all array with train, valid and test sets
    Returns:
        d-grams: dictionary of ngrams and counts of each row
        whole_grams: dictionary with accumulated count
    '''
    whole_grams = {}
    d_grams = {}
    for didx, txt in enumerate(txt_arr):
        grams, whole_grams = ngram_creater(n, txt, whole_grams)
        d_grams[didx] = grams

    return d_grams, whole_grams


def word_to_index(n, txt_arr):
    '''
    The function assigns index to each unique words
    or n-grams.
    Inputs:
        n: n for n-grams
        txt_arr: all array with train, valid and test sets
    Returns:
        word_to_idx: dictionary mapping word to index
        wtorch: dictionary mapping word to counts
    '''
    count = 0
    word_to_idx = {}
    _, whole_grams = word_ngrams(n, txt_arr)
    wtorch = torch.zeros(len(whole_grams))
    
    for ngrams, ct in whole_grams.items():
        word_to_idx[ngrams] = count
        wtorch[count] = ct
        count += 1
    
    return word_to_idx, wtorch


def sentence_torch(grams, word_to_idx, wtorch):
    '''
    The function transfers a list of phrase to an
    embedding vector.
    Inputs:
        grams: list of splitted sentence
        word_to_idx: dictionary mapping word to index
        wtorch: dictionary mapping word to counts
    Return: an embedding vector 
    '''
    bow_vec = torch.zeros(len(word_to_idx))
    for gram in grams:
        bow_vec[word_to_idx[gram]] = wtorch[word_to_idx[gram]]
        
    return bow_vec


def make_target(label, label_to_ix):
    '''
    Helper function for label tensor creation
    '''
    return torch.LongTensor([label_to_ix[label]])


def run_bow_ngram(model, word_to_idx, wtorch, X_train, y_train, 
                  X_valid, y_valid, X_test, y_test, n):
    '''
    The function is used to run the neural network classifier
    with ngrams and bag of words setting
    Inputs:
        model: model objects for neural network
        word_to_idx: word to indices dictionary 
        wtorch: bag of words tensor
        X_train, X_train, y_train, X_valid, y_valid, X_test, y_test
        n: n for ngrams
    Returns: the performance dictionary
    '''
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    best_acc = 0
    perf_dict = {}
    for epoch in range(5):
    
        for idx in range(X_train.shape[0]):
        
            model.zero_grad()
        
            sentence = X_train[idx]
            grams, _ = ngram_creater(n, sentence, {})
            bow_vec = sentence_torch(grams, word_to_idx, wtorch)
        
            pred = model(bow_vec.view(1,-1))
            loss = loss_function(pred, make_target(y_train[idx, 0], label_to_ix))
            loss.backward()
            optimizer.step()
    
        acc_count = 0
        with torch.no_grad():
            for idx in range(X_valid.shape[0]):
                sentence = X_valid[idx]
                grams, _ = ngram_creater(n, sentence, {})
                bow_vec = sentence_torch(grams, word_to_idx, wtorch)
                pred = model(bow_vec.view(1,-1))

                y_pred = np.argmax(pred[0].detach().numpy())
                if y_valid[idx, 0] == y_pred:
                    acc_count += 1

        print("For epoch number ", epoch, ", the accuracy for validation set is ", 
              acc_count / X_valid.shape[0])
    
        if (acc_count / X_valid.shape[0]) > best_acc:
            best_model = model
        
        perf_dict["valid_epoch_acc " + str(epoch)] = acc_count / X_valid.shape[0]

    acc_count = 0
    yreal_count = 0
    ypred_count = 0
    ypred1_count = 0
    with torch.no_grad():
        for idx in range(X_test.shape[0]):
            sentence = X_test[idx]
            grams, _ = ngram_creater(n, sentence, {})
            bow_vec = sentence_torch(grams, word_to_idx, wtorch)
            pred = best_model(bow_vec.view(1,-1))
            y_pred = np.argmax(pred[0].detach().numpy())

            if y_test[idx, 0] == y_pred:
                acc_count += 1
        
            yreal_count += y_test[idx, 0]
            ypred_count += y_pred
        
            if (y_test[idx, 0] == y_pred) and (y_pred == 1):
                ypred1_count += 1

    print("the accuracy for test set is ", acc_count / X_test.shape[0])
    print("the presision for test set is ", ypred1_count / ypred_count)
    print("the recall for test set is ", ypred1_count / yreal_count)

    perf_dict["best_test_acc"] = acc_count / X_test.shape[0]
    perf_dict["best_test_precision"] = ypred1_count / ypred_count
    perf_dict["best_test_recall"] = ypred1_count / yreal_count

    return perf_dict