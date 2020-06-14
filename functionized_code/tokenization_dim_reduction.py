import pandas as pd
import torch
import torch.utils.data as tud
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.feature_extraction.text as FE
from collections import Counter
import os
import numpy as np
import random
import copy
import re


# Import Data

cols_t1 = ["video_id", "title", "category_id"]
cols_t2 = ["video_id", "tags", "category_id"]
cols_t3 = ["video_id", "title", "tags", "category_id"]
cols_t4 = ["video_id", "title", "tags", "description", "category_id"]

def select_col(data_dir, cols):
    '''
    The function is used to extract matrices of
    video id, text features and labels
    Input:
        data_dir: directory of data storage
        cols: columns selected
    Return: video indices, text matrix, label vector
    '''
    data = pd.read_csv(data_dir)[cols]
    data = data.drop_duplicates("video_id", "first")
    sub_data = np.array(data)
    idx = sub_data[:, 0]
    labels = sub_data[:, -1]
    
    text_data = sub_data[:, 1:-1]

    return idx, text_data, labels  

def multi_to_binary(labels, target_label):
    '''
    The helper function is used to transfer
    multi-class to binary class
    '''
    new_labels = np.zeros([len(labels), 1])
    new_labels[labels == target_label] = 1

    return new_labels


# Preprocessing Text

def tokenized_tags(tag):
    '''
    The helper function to exclude delimiter in tags
    '''
    single_tag = ""
    for tag in tag.split("|"):
        single_tag = single_tag + " " + tag
    
    return single_tag


def combine_text(text_arr, tag_idx, other_idx):
    '''
    The function merges the text columns to a
    single text column
    Input:
        text_arr: text array of all text columns
        tag_idx: the column index for tags
        other_idx: the remaining indices
    Return: combined text vector
    '''
    final_arr = text_arr[:, 0]
    for idx in other_idx[1:]:
        final_arr = final_arr + " " + text_arr[:, idx].astype(str)
    
    if tag_idx != -1:
        new_lst = []
        for tag in text_arr[:, tag_idx]:
            comb_tag = tokenized_tags(tag)
            new_lst.append(re.sub('"' ,"", comb_tag))
        new_tags = np.array(new_lst)

        if other_idx == []:
            return new_tags
        else:
            return final_arr + new_tags
    
    return final_arr


# Tokenization

def tfidf_tokenization(arr):
    '''
    The function tokenized embedding matrix
    with TFIDF method
    Input: embedding matrix or array
    Return: tokenized matrix
    '''
    vectorizer = FE.TfidfVectorizer()
    X = vectorizer.fit_transform(arr)
    
    return X


# Dimensional Reduction and LSI

def dimensional_reduction(df, k, get_test_df=False, test_df=""):
    '''
    The function is designed to make dimensional reduction with SVD method
    Inputs:
        df: original matrix
        k: the number of singular values taken
        y: label vector
        allm: if True, return the components of economic SVD
        get_weight: if True, return the weight calculated by the economic SVD
                    components
    Returns: approximated df with k singular values
    '''
    U_k = np.linalg.svd(df)[0][:, :k]
    sigma_k = np.linalg.svd(df)[1][:k]
    Vt_k = np.linalg.svd(df)[2][:k, :]

    reduced_df = (Vt_k[:k, :].dot(df.T)).T

    if get_test_df:
        return reduced_df, (Vt_k[:k, :].dot(test_df.T)).T
    
    return reduced_df





