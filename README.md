# youtube-labels
Ta-Yun Yang & Patrick Lavallee Delgado <br>
Candidates, MS Computational Analysis and Public Policy

## Final presentation slides

- [Slide deck](presentation/slides.pdf)

## Recovering Self-selected YouTube Video Categories
We attempt to recreate the decision a YouTube user makes selecting a category for his video using other instances of self-expression in the same content. These observations are encoded in language, including the title, description, tags, and caption of the video. This repository documents the data and modeling pipeline we use in our analysis.

## Requirements
- altair 4.1.0
- numpy 1.16.4      
- pandas 1.0.4
- PyYAML 5.1.1 
- scikit-learn 0.21.3
- spacy 2.2.4
- torch 1.4.0       
- torchtext 0.6.0
- youtube-transcript-api 0.3.1

## Getting started
The program `run_pipeline.py` executes the data and modeling pipelines. The `--config` flag identifies the location of the `YAML` file that parameterizes the pipeline. The `--out` flag identifies the directory in which to save the results of the models and the model itself.
```
$ python3 run_pipeline.py --config models_w_captions/unigrams.yaml --out models_w_captions
```
The `YAML` file specifies the columns that correspond to the labels and corpus in the data for the purposes of a particular experiment. It gives flexibility to specify which text features to include in the corpus; for example, we use this to run our models with and without the caption data. Similarly, the file sets which categories in the label to include and on which to binarize. It also sets the n-gram, vocabulary, and batch sizes.

The same file parameterizes the models to run on that configuration of data. Each model takes the name and initialization parameters from its class definition in `nnmodels.py`. Notice that the input dimension and index of the padding token are not specified here since these value rely on the results that come from the data pipeline. Here is a short example:
```
data:
  col_labels: 'category_id'
  col_corpus: ['title', 'tags', 'description', 'caption']
  label_target: 'News & Politics'
  label_others: ['Entertainment']
  splits: [0.4, 0.4, 0.2]
  ngram_size: 1
  vocab_size: 25000
  batch_size: 64
models:
  - model: CNN
    embedding_dim: 100
    n_filters: 200
    filter_sizes: [3, 4, 5]
    output_dim: 1
    dropout: 0.5
  - model: LSTM
    embedding_dim: 100
    hidden_dim: 50
    output_dim: 1
    n_layers: 4
    bidirectional: True
    dropout: 0.5
decision_metric: 'loss'
out_directory: 'politics_entertainment/'
```

The notebooks in the repository are deeper investigations into the data and models. The section that follows describes each.

## Repository structure
- `data/`: video and category data by country as well as caption data for videos in the US data.
- `get_captions.py`: scrape the caption data from the video page on the YouTube website (47 lines, Patrick).
- `exploration.ipynb`: notebook that collects the counts of videos by category and the intersections of their vocabularies (105 lines, Patrick).
- `models_w_captions/`: results of models on data with captions.
- `models_wo_captions/`: results of models on data without captions.
- `run_pipeline.py`: program that runs models on the data per specification by a `YAML` configuration (380 lines, Patrick).
- `functionized_code`:
    - `data_pipeline.py`: select labels and text features from the data, build the vocabulary from the corpus, create vector representations of each observation in the corpus, split the data into training, validation, and testing sets (506 lines, Patrick).
    - `data_input.py`: functions used to import and preprocess YouTube data (including Titles / Tags / Descriptions and Captions) (222 lines, Ta-Yun).
    - `tokenization_dim_reduction.py`: functions used to operate TFIDF tokenization and dimensional reductions (133 lines, Ta-Yun).
    - `nnmodels.py`: Neural Network Model objects (including Linear NN, CNN, RNN) (192 lines, Ta-Yun).
    - `mlmodels.py`: functions used to process the analysis with conventional models (112 lines, Ta-Yun).
    - `bow_models.py`: functions used transform data to Bag of Words structure and process linear model (188 lines, Ta-Yun).
    - `ngrams.py`: functions used to preprocess language data and create ngrams (41 lines, Patrick).
    - `train_eval.py`: functions used to process the analysis of neural network models (218 lines, Ta-Yun & Patrick).
    - `cosine_similarity.py`: functions used to calculate cosine similarities and conduct bottom-up clustering (274 lines, Ta-Yun).
    - `visualization.py`: functions used to implement data visualization for model comparison (88 lines, Ta-Yun).
    - `run_nn_models_politics.ipynb`: report of politics v.s. non-politics classification using titles/ tags / description data (173 lines, Ta-Yun).
    - `run_nn_models_entertainment.ipynb`: report of entertainment v.s. non-entertainment classification using titles/ tags / description data (173 lines, Ta-Yun).
    - `run_captions_model.ipynb`: report of NN classification using captions data (170 lines, Ta-Yun).
    - `run_captions_model_GRU.ipynb`: report of GRU and ML classification using captions data (222 lines, Ta-Yun).
    - `test_nn_sub-category.ipynb`: report of binary category classification using titles/ tags / description data (170 lines, Ta-Yun).
    - `similarity_test.ipynb`: report of similarity tests and bottom-up clustering (135 lines, Ta-Yun).
- `presentation/`: figures in the slide deck.
- `proposal/`: research and writeup that oriented this project.
