import bow_models as bowm
from torchtext import data
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import tokenization_dim_reduction as tdr
import sklearn.feature_extraction.text as FE

COS = nn.CosineSimilarity(dim=0)
POL_ID = 25

def get_category_arr(new_pd, politics_idx, other_idx):
    '''
    The function subsets the dataframe with two category indices
    Inputs:
        new_pd: input dataframe
        politics_idx: index for political category
        other_idx: index for other category used
    Returns: sub-dataframes with political and other categories respectively
    '''
    politics_arr = np.array(new_pd[new_pd["category_id"] == politics_idx]["text"])
    other_arr = np.array(new_pd[new_pd["category_id"] == other_idx]["text"])

    return politics_arr, other_arr


def similarity_bow_pair(new_pd, cos, politics_idx, other_idx, k):
    '''
    The function calculates the cosine similarity using Bag of Words
    method from two vectors representing two categories
    Inputs:
        new_pd: input dataframe
        cos: cosine function object
        politics_idx: index for political category
        other_idx: index for other category used
        k: keep k highest BOW counts for each vector
    Return: cosine similarity value
    '''
    pol, oth = get_category_arr(new_pd, politics_idx, other_idx)
    pols = " ".join(pol)
    oths = " ".join(oth)
    word_to_index, wtorch = bowm.word_to_index(1, np.array(new_pd["text"]))
    
    pol_grams, _ = bowm.ngram_creater(1, pols, {})
    oth_grams, _ = bowm.ngram_creater(1, oths, {})
    pol_bow = bowm.sentence_torch(pol_grams, word_to_index, wtorch)
    oth_bow = bowm.sentence_torch(oth_grams, word_to_index, wtorch)
    
    top5k_idx = wtorch.topk(k).indices
    similarity = cos(pol_bow[top5k_idx], oth_bow[top5k_idx]).item()
    
    return similarity


def similarity_bow(dic_dir, new_pd, cos, politics_idx, k):
    '''
    The function calculates the cosine similarity using Bag of Words
    method between political and each of other category
    Inputs:
        dic_dir: the directory of the category dictionary
        new_pd: input dataframe
        cos: cosine function object
        politics_idx: index for political category
        k: keep k highest BOW counts for each vector
    Returns:
        cat_dict: dictionary of the category indices and names
        cat_size: dictionary of the category and its corresponding number of rows
        similarities: cosine similarity values
    '''
    with open(dic_dir) as f:
        cat_ids = json.load(f)

    cat_dict = {}
    for cat_it in cat_ids["items"]:
        cat_dict[cat_it["id"]] = cat_it["snippet"]["title"]

    similarities = {}
    for cat_id, name in cat_dict.items():
        similarities[name] = similarity_bow_pair(new_pd, cos, 
                                                 politics_idx, 
                                                 int(cat_id), k)
    cat_size = {}
    for i, val in cat_dict.items():
        cat_size[(i, val)] = new_pd[new_pd["category_id"] == int(i)].shape[0]
    
    return cat_dict, cat_size, similarities


def similarity_tfidf_pair(new_pd, cos, pol_idx, oth_idx, ks):
    '''
    The function calculates the cosine similarity using TFIDF
    method from two vectors representing two categories
    Inputs:
        new_pd: input dataframe
        cos: cosine function object
        politics_idx: index for political category
        other_idx: index for other category used
        ks: keep k[0] to k[1] highest mean TFIDF value for each vector
    Return: cosine similarity value
    '''
    new_arr = np.array(new_pd["text"])
    vectorizer = FE.TfidfVectorizer()
    txt_tfidf = vectorizer.fit_transform(new_arr)
    top5kidx = np.argsort(txt_tfidf.toarray().mean(0))[-ks[1]:-ks[0]]
    feature_names = vectorizer.get_feature_names()
    
    pol_arr, oth_arr = get_category_arr(new_pd, pol_idx, oth_idx)
    pol_tfidf = vectorizer.transform(pol_arr).toarray()
    oth_tfidf = vectorizer.transform(oth_arr).toarray()
    
    pol_mvec = pol_tfidf[:, top5kidx].mean(0)
    oth_mvec = oth_tfidf[:, top5kidx].mean(0)

    return cos(torch.tensor(pol_mvec), torch.tensor(oth_mvec)).item()


def similarity_tfidf(cat_dict, cat_size, new_pd, cos, pol_idx, ks):
    '''
    The function calculates the cosine similarity using TFIDF
    method between political and each of other category
    Inputs:
        cat_dict: dictionary of the category indices and names
        cat_size: dictionary of the category and its corresponding number of rows
        new_pd: input dataframe
        cos: cosine function object
        politics_idx: index for political category
        ks: keep k[0] to k[1] highest mean TFIDF value for each vector
    Return: cosine similarity value
    '''
    similarities = {}
    for cat_id, name in cat_dict.items():
        if cat_size[(cat_id, name)] != 0:
            similarities[name] = similarity_tfidf_pair(\
                       new_pd, cos, pol_idx,int(cat_id), ks)
        else:
            similarities[name] = 0

    return similarities


class category_node:
    '''
    Tree Node for categories which store information for the similarity score
    '''
    def __init__(self, name, categories, childrens):
        
        self.name = name
        self.categories = categories
        self.childrens = childrens
        self.performances = {}
        self.similarities = {}
        self.next_id = None
        self.next_name = None
        self.best_similarity = None
    
    def get_similarities(self, cat_dict, cat_size, new_pd, pol_id, tfidf_range):
        '''
        The function calculates the similarity scores between political
        vector and each of other category. For each turn, get the highest
        similarity and the corresponding name and index
        Inputs:
            cat_dict: dictionary of the category indices and names
            cat_size: dictionary of the category and its corresponding number of rows
            new_pd: input dataframe
            pol_idx: index for political category
            tfidf_range: keep k[0] to k[1] highest mean TFIDF value for each vector
        '''
        cat_to_idx = {}
        for idx, name in cat_dict.items():
            cat_to_idx[name] = idx

        tfidf_similarities = similarity_tfidf(cat_dict, cat_size, new_pd, COS, pol_id, tfidf_range)
        self.similarities = tfidf_similarities
        
        max_sim = 0
        for cat_name, sim in tfidf_similarities.items():
            if ((sim != 0) and (sim != 1) and (sim > max_sim)):
                max_sim = sim
                next_cat = cat_name
                next_id = cat_to_idx[cat_name]
        
        self.next_id = next_id
        self.next_name = next_cat
        self.best_similarity = max_sim


def bottom_up(cat_dict, cat_size, new_pd, tfidf_range):
    '''
    The function calculates the bottom-up clustering using TFIDF
    method between political and each of other category
    Inputs:
        cat_dict: dictionary of the category indices and names
        cat_size: dictionary of the category and its corresponding number of rows
        new_pd: input dataframe
        tfidf_range: keep k[0] to k[1] highest mean TFIDF value for each vector
    Return: nodes of similarities
    '''
    cat_to_idx = {}
    for idx, name in cat_dict.items():
        cat_to_idx[name] = idx

    init_similarities = similarity_tfidf(cat_dict, cat_size, new_pd, COS, POL_ID, tfidf_range)
    tree_height = 0
    nonzero_similarities = {}
    for name, sim in init_similarities.items():
        if sim > 0:
            nonzero_similarities[name] = sim
            tree_height += 1
    
    count = 0
    new_cat_dict = {}
    for idx, name in cat_dict.items():
        if init_similarities[name] != 0:
            new_cat_dict[idx] = name
    cat_dict = new_cat_dict

    pol_node = category_node(cat_dict[str(POL_ID)], [POL_ID], [])

    while count < tree_height - 1:
        pol_node.get_similarities(cat_dict, cat_size, new_pd, POL_ID, tfidf_range)
        new_pd.loc[new_pd["category_id"] == int(pol_node.next_id), "category_id"] = POL_ID
    
        new_name = pol_node.name + " (and) " + pol_node.next_name
        nxt_node = category_node(pol_node.next_name, int(pol_node.next_id), [])
        pol_node = category_node(new_name, pol_node.categories + [int(pol_node.next_id)], [pol_node, nxt_node])
    
        new_cat = {}
        for idx, name in cat_dict.items():
            if name != nxt_node.name:
                new_cat[idx] = name
        cat_dict = new_cat
    
        count += 1
    
    return pol_node


def get_topk_words(new_pd, k):
    '''
    The function extract the top k words with the highest TFIDF 
    that all documents in each categories are merged into single document
    Inputs:
        new_pd: input dataframe
        k: top k words with the highest TFIDF
    Return: dictionary with each category and its corresponding top k words
    '''
    txts = []
    labels = []
    idx_dict = {}
    for i, cat in enumerate(pd.unique(new_pd["category_id"])):
        idx_dict[cat] = i
        sub_df = new_pd[new_pd["category_id"] == cat]
    
        sub_txt = ""
        for txt in sub_df["text"]:
            sub_txt = sub_txt + " " + txt
        txts.append(sub_txt)
        labels.append(cat)

    combine_df = pd.DataFrame(data={"text": txts, "category_id":labels})
    vectorizer = FE.TfidfVectorizer(analyzer=u'word', max_df=0.95, lowercase=True, 
                                    stop_words="english", max_features=25000)
    combined_tfidf = vectorizer.fit_transform(np.array(combine_df["text"]))
    feature_names = vectorizer.get_feature_names()

    full_dict = {}
    for cat in pd.unique(new_pd["category_id"]):
        sub_tfidf = combined_tfidf[idx_dict[cat], :].toarray()
        sub_top100 = np.argsort(sub_tfidf)[0][-k:]
        full_dict[cat] = np.array(feature_names)[list(sub_top100)]
    
    return full_dict