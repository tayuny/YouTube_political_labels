import numpy as np
import pandas as pd
import altair as alt

def process_perf_dict(models_perf, mlm_perform, perf_dict, epoch):
    '''
    The function is used to pre-process the performance dictionary
    for visualization
    Inputs:
        models_perf: dictionary of neural network model performances
        mlm_perform: dictionary of machine learning model performances
        perf_dict: dictionary of ngrams-BOW model performances
        epoch: number of epoches
    Return: datasets for performance visualizations
    '''
    models_perf["ngram_bow"] = perf_dict
    for key, tuples in mlm_perform.items():
        models_perf[key] = {}
        models_perf[key]["best_test_acc"] = tuples[1]
        models_perf[key]["best_test_precision"] = tuples[2]
        models_perf[key]["best_test_recall"] = tuples[3]
    perf_df = pd.DataFrame(models_perf)

    select_cols = ["avg_embedding", "SimpleRNN", "BLSTM", "CNN", "ngram_bow"]
    epoch_df = perf_df[select_cols].iloc[:epoch]
    full_perf_df = pd.DataFrame(columns=["index", "accuracy", "model"])
    for col in epoch_df.columns:
        sub_df = epoch_df[col].reset_index()
        sub_df = sub_df.rename(columns={col : "accuracy"})
        sub_df["model"] = col
        full_perf_df = pd.concat([full_perf_df, sub_df], join="inner")

    best_perfs = perf_df.iloc[epoch:]
    best_cols = ["model", "best_test_acc", "best_test_precision", "best_test_recall"]
    full_perf_df2 = pd.DataFrame(columns = best_cols)
    for col in best_perfs.columns:
        sub_arr = pd.DataFrame([[col] + list(best_perfs[col])], columns = best_cols)
        full_perf_df2 = pd.concat([full_perf_df2, sub_arr])

    layer_perf_cols = ["model", "eva_value", "eval"]
    layer_perf_df = pd.DataFrame(columns=layer_perf_cols)
    for eva in best_cols[1:]:
        full_perf_df2 = full_perf_df2.reset_index(drop=True)
        sub_df = full_perf_df2[["model", eva]]
        sub_df = sub_df.rename(columns={eva: "eva_value"})
        sub_df["eval"] = eva
        layer_perf_df = pd.concat([layer_perf_df, sub_df], join="inner")

    return full_perf_df, full_perf_df2, layer_perf_df 


def plot_nn_performance(full_perf_df):
    '''
    The function is used to create the visualization
    of model accuracy of each epoch
    Inputs: model performance datasets
    Returns: the plot of Model Accuracy for Each Epoch
    '''
    plt = alt.Chart(full_perf_df).mark_line().encode(
        x='index',
        y=alt.Y('accuracy', scale = alt.Scale(domain=[0.6, 1])),
        color=alt.Color('model', scale=alt.Scale(scheme="tableau20"))
    ).properties(
        width=500,
        height=200,
        title="Model Accuracy for Each Epoch"
    )
    return plt


def plot_best_model(layer_perf_df):
    '''
    The function is used to create the visualization
    of best evaluation for each model
    Inputs: model performance datasets
    Returns: the plot of Model Evaluation for Each Model
    '''
    plt = alt.Chart(layer_perf_df).mark_bar().encode(
        x='eval',
        y=alt.Y('eva_value'),
        color=alt.Color('eval', scale=alt.Scale(scheme="tableau20")),
        column="model"
    ).properties(
        width=100,
        height=400,
        title="Model Evaluation for Each Model"
    )
    return plt