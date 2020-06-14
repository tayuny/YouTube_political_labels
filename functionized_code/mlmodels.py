import numpy as np
import pandas as pd
import random
import copy
import torch
import tokenization_dim_reduction as tdr
import ngrams as ng
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

PARAMETER_DICT = {"random_forest":{'n_estimators': [1, 10,100], 'max_depth': [5,50],
                              'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
                  "logistics": {'penalty': ["l1",'l2'], 'C': [0.001,0.1,1]}}

CLFS = {'random_forest': RandomForestClassifier(n_jobs=2),
        'logistics': LogisticRegression(C=1e5, solver='liblinear')}


def classifier_developer(method, clfs, parameter_dict):
    '''
    This function is used to generate possible combination of hyperparameters of
    given classifiers
    Inputs:
        method: specific classifiers
        clfs: dictionary of classifiers
        parameter_dict: parameters dictionary for the classifiers
    Returns: list of all possible combination of parameters
    '''
    parameters = parameter_dict[method]
    para_list = ParameterGrid(parameters)

    return para_list


def eval_acc(y_pred, y_real):
    '''
    The helper function to evaluate the accuracy
    of true label and predicted probability
    '''
    pdy1 = np.zeros(len(y_pred))
    pdy1[y_pred >= 0.5] = 1
    pdy2 = np.zeros(len(y_pred))
    pdy2[pdy1 == y_real[:, 0]] = 1
    
    return sum(pdy2) / len(pdy2)


def eval_precision_recall(y_pred, y_real):
    '''
    The helper function to evaluate the precision
    and recall of true label and predicted probability
    '''
    pdy1 = np.zeros(len(y_pred))
    pdy1[y_pred >= 0.5] = 1

    pdy2 = np.zeros(len(y_pred))
    
    if len(pdy1[pdy1 == 1]) == 0:
        precision = 0
    else:
        precision = len(pdy1[(pdy1 == 1) & (pdy1 == y_real[:, 0])]) / \
                    len(pdy1[pdy1 == 1])

    recall = len(pdy1[(pdy1 == 1) & (pdy1 == y_real[:, 0])]) / \
             len(pdy1[y_real[:, 0] == 1])

    return precision, recall


def ml_evaluate(clfs, parameter_dict, X_train, y_train, X_valid, y_valid,
                X_test, y_test):
    '''
    The function evaluated the performance of each machine learning
    model type and with the hyper-parameters defined in the parameter-
    dict. The best accuracy of each model type is presented in the
    output.
    Inputs:
        clfs: dictionary of classifiers
        parameter_dict: dictionary of parameters for each classifier
        X_train, y_train, X_valid, y_valid, X_test, y_test
    Returns: dictionary with model type, parameters and best accuracy
    '''
    outputs_dict = {}
    for method, _ in clfs.items():
        if method in parameter_dict:
            print("operation of {} method begins".format(method))
            para_list = classifier_developer(method, clfs, parameter_dict)
            best_acc = 0
            for para in para_list:
                clf = clfs[method].set_params(**para)
                model_name = method + " with parameters : " + str(clf.get_params())
                model = clf.fit(X_train, y_train[:, 0])
                
                y_v_predp = model.predict_proba(X_valid)[:, 1]
                y_t_predp = model.predict_proba(X_test)[:, 1]
                acc_v = eval_acc(y_v_predp, y_valid)
                acc_t = eval_acc(y_t_predp, y_test)
                test_prec, test_rec = eval_precision_recall(y_t_predp, y_test)

                if acc_v > best_acc:
                    best_acc = acc_t
                    best_prec = test_prec
                    best_rec = test_rec
                    best_model = model_name
            
            outputs_dict[method] = (best_model, best_acc, test_prec, test_rec)
            print("the best accuracy for method: ", method,  " is ", best_acc)
            print("the corresponding precision is : ", best_prec)
            print("the corresponding recall is : ", best_rec)

    return outputs_dict