from typing import Tuple, List, Dict, Set, Any
from functools import wraps
import re
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction import stop_words


# Identify constants for vocabulary operations.
DELIMITERS = re.compile(r' |\\n|\|')
PUNCTUATION = re.compile(r'[.:;,?!\"|#()-_•]|^\'|\'$')
STOP_WORDS = stop_words.ENGLISH_STOP_WORDS
STOP_PATTERN = re.compile(r'http|www|^\s*$')
UNK = '<unk>'
PAD = '<pad>'
DEFAULT_KWARGS = {
    'tokenize_pattern': DELIMITERS,
    'clean_pattern': PUNCTUATION,
    'stop_words': STOP_WORDS,
    'drop_pattern': STOP_PATTERN
}


# Write a decorator to add default keyword arguments.
def include_default_kwargs(**default_kwargs):
    def decorator(f):
        @wraps(f)
        def g(*args, **kwargs):
            default_kwargs.update(kwargs)
            return f(*args, **default_kwargs)
        return g
    return decorator


class TorchTextLike():
    '''
    Represent a label, sentence pair in a container that handles like an
    Example object from the torchtext.data library. These attributes are
    those expected in train_eval.py.
    '''

    # Identify the device on which to put tensors.
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __init__(self, label: torch.Tensor, text: torch.Tensor):
        self.label = label.to(device=TorchTextLike.DEVICE)
        self.text = text.to(device=TorchTextLike.DEVICE)


def get_labels_and_corpus(
    data: pd.DataFrame,
    idx_labels: int,
    idx_corpus: List[int],
    label_target: Any = None,
    labels_other: List[Any] = None,
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Extract the labels and corpus identified in a dataframe. Concatenate the
    columns given for the corpus, keep only observations with select labels if
    any indicated, and binarize on the target label.

    data (pd.DataFrame): original data to parse.
    idx_labels (int): index at which to find labels.
    idx_corpus (list of ints): indices at which to find the corpus.
    label_target (Any): the positive class on which to binarize labels.
    labels_other (list of Any): any other classes to include with labels.

    Return labels and corpus (tuple of np.ndarray).
    '''
    # Identify observations that belong to either the target or other classes.
    if labels_other:
        keep = [label_target] + labels_other if label_target else labels_other
        idx_keep = data.index[data.iloc[:, idx_labels].isin(keep)]
    else:
        idx_keep = np.arange(data.shape[0])
    # Extract the labels and corpus from the data.
    labels = data.iloc[idx_keep, idx_labels]
    corpus = data.iloc[idx_keep, idx_corpus]
    # Concatenate text features in the corpus.
    if len(idx_corpus) > 1:
        corpus = corpus.astype(str).agg(' '.join, axis=1)
    # Binarize on the target label if requested.
    if label_target:
        labels = labels.apply(lambda y: 1 if y == label_target else 0)
    # Report on the data.
    if not silent:
        __show_label_stats(labels)
    # Return the labels and corpus.
    return labels.to_numpy(), corpus.to_numpy()


@include_default_kwargs(**DEFAULT_KWARGS)
def get_vocab_and_batches(
    labels: np.ndarray,
    corpus: np.ndarray,
    splits: Tuple[float, float, float],
    ngram_size: int,
    vocab_size: int,
    batch_size: int,
    cbow: bool = False,
    silent: bool = False,
    **kwargs: dict
) -> Tuple[Tuple[Dict[str, int], Dict[int, str]], List[List[TorchTextLike]]]:
    '''
    Preprocess the data for modeling with a neural network. Split the data into
    training, validation, and testing sets; get the vocabulary of n-grams from
    the training set; get batches of sentence vectors in each set, represented
    as either simple padded numericizations or continuous bags of words.

    labels (np.ndarray): 1-d array of labels.
    corpus (np.ndarray): 1-d array of sentences to parse.
    splits (tuple of floats): proportional allocations of each respective set.
    ngram_size (int): length of n-gram, i.e. number of words in one token.
    vocab_size (int): number of most frequent words to keep.
    batch_size (int): number of sentences with each batch.
    cbow (bool): whether to represent sentences as continuous bags of words.
    kwargs (dict): keyword arguments to pass to __make_sentence().

    Return vocab (tuple of dicts), batches (list of lists of TorchTextLike).
    '''
    # Split the data into training, validation, and testing sets.
    datasets = get_splits(labels, corpus, splits)
    # Get the vocabulary from the training set.
    _, train_corpus = datasets[0]
    vocab = get_vocab(train_corpus, ngram_size, vocab_size, **kwargs)
    # Get batches for each dataset.
    dataset_batches = []
    for labels, corpus in datasets:
        batches = get_batches(
            vocab[0], labels, corpus, ngram_size, batch_size, cbow, **kwargs)
        dataset_batches.append(batches)
    # Report on the vocabulary and batches.
    if not silent:
        __show_vocab_stats(vocab[0])
        __show_batch_stats(dataset_batches)
    # Return the vocabulary and batches.
    return vocab, dataset_batches


def get_splits(
    labels: np.ndarray,
    corpus: np.ndarray,
    splits: Tuple[float]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    '''
    Split the data into training, validation, and testing sets.
    
    labels, corpus (np.ndarray): data to split.
    splits (tuple of floats): proportional allocations of each respective set.

    Return training, validation, testing sets (list of tuples of np.ndarray).
    '''
    datasets = []
    # Ensure the size of the labels and corpus match.
    len_corpus = corpus.shape[0]
    assert labels.shape[0] == corpus.shape[0]
    # Slice and copy the data with indices that allocate each set exclusively.
    for idx in __make_splits(len_corpus, *splits):
        dataset = np.copy(labels[idx]), np.copy(corpus[idx])
        datasets.append(dataset)
    # Return the training, validation, testing sets.
    return datasets


@include_default_kwargs(**DEFAULT_KWARGS)
def get_vocab(
    corpus: np.ndarray,
    n: int,
    k: int,
    **kwargs: dict
) -> Tuple[Dict[str, int], Dict[int, str]]:
    '''
    Get the k most frequent words in the vocabulary of n-grams as mappings of
    words to indices and indices to words.

    corpus (np.ndarray): 1-d array of sentences to parse.
    n (int): length of n-gram, i.e. number of words to represent with one token.
    k (int): number of most frequent words to keep.
    kwargs (dict): keyword arguments to pass to __make_sentence().

    Return mappings of the vocabulary (tuple of dicts).
    '''
    # Collect the vocabulary in the corpus.
    vocab = __make_vocab(corpus, n, **kwargs)
    # Collect the mapping of the k most frequent words to their indices.
    word_to_idx = __make_index(vocab, k)
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return word_to_idx, idx_to_word


@include_default_kwargs(**DEFAULT_KWARGS)
def get_batches(
    vocab: Dict[str, int],
    labels: np.ndarray,
    corpus: np.ndarray,
    n: int,
    k: int,
    cbow: bool = False,
    **kwargs: dict
) -> List[TorchTextLike]:
    '''
    Get k batches of sentences in the corpus as their vector representations, 
    either simple padded numericizations or continuous bags of words.

    vocab (dict): mapping of words to indices.
    labels (np.ndarray): 1-d array of labels.
    corpus (np.ndarray): 1-d array of sentences to parse.
    n (int): length of n-gram, i.e. number of words to represent with one token.
    k (int): number of sentences with each batch.
    cbow (bool): whether to represent sentences as continuous bags of words.
    kwargs (dict): keyword arguments to pass to get_vectorization().

    Return batches of label, sentence pairs (list of tuples).
    '''
    batches = []
    # Ensure the size of the labels and corpus match.
    len_corpus = corpus.shape[0]
    assert len_corpus == labels.shape[0]
    # Vectorize sentences.
    sentences = [
        __make_vector(vocab, corpus[idx], n, cbow, **kwargs)
        for idx in range(len_corpus)
    ]
    # Sort sentences by length, descending, to minimize padding in each batch.
    sentences.sort(key=lambda x: x.shape[0], reverse=True)
    # Get the type of which to make the batches; embeddings require long type.
    dtype = torch.float if cbow else torch.long
    # Calculate the number of batches to create.
    len_batches = (len_corpus + k - 1) // k
    # Create each batch.
    for i in range(len_batches):
        # Identify the indices in the data that correspond to this batch.
        idx_min = i * k
        idx_max = min(idx_min + k, len_corpus)
        # Batch sentences and labels.
        batch_X = __make_batch(sentences[idx_min:idx_max], vocab[UNK], dtype)
        batch_y = torch.as_tensor(
            labels[idx_min:idx_max], dtype=torch.float).squeeze()
        # Ensure the text and label tensors have the same length.
        assert batch_X.shape[1] == batch_y.shape[0]
        # Include this batch with all batches.
        batches.append(TorchTextLike(batch_y, batch_X))
    # Return the batches.
    return batches


def __make_splits(
    n: int,
    train: float,
    valid: float,
    test: float
) -> Tuple[np.ndarray]:
    '''
    Get the indices that proportionally allocate training, validation, and
    testing sets.

    n (int): length of the data.
    train, valid, test (float): proportional allocations of each respective set.

    Return indices of each split (tuple of np.ndarray).
    '''
    # Ensure the proportions sum to one.
    assert sum([train, valid, test]) == 1
    # Allocate a random permutation of row indices proportionally.
    idx = np.random.permutation(n)
    idx_train = idx[:int(n * train)]
    idx_valid = idx[int(n * train):int(n * (train + valid))]
    idx_testg = idx[int(n * (train + valid)):]
    # Return the subsets of data.
    return idx_train, idx_valid, idx_testg


def __make_vocab(
    corpus: np.ndarray,
    n: int,
    **kwargs: dict
) -> Dict[str, int]:
    '''
    Collect the vocabulary in the corpus and tally the frequency of each word.
    Identify words on the delimiter and clean them with a regular expression.

    corpus (np.ndarray): 1-d array of sentences to parse.
    n (int): length of n-gram, i.e. number of words to represent with one token.
    kwargs (dict): keyword arguments to pass to __make_sentence().

    Return mapping of words to frequencies (dict).
    '''
    vocab = {}
    # Consider each sentence in the corpus.
    for sentence in corpus:
        # Preprocess the sentence, i.e. tokenize, clean, drop, pad.
        sentence = __make_sentence(sentence, n, **kwargs)
        # Consider each word, or possibly n-gram, in the sentence.
        for w in range(len(sentence) - n):
            # Collect this word or n-gram.
            word = ' '.join(sentence[w:w + n])
            # Update the frequency of this word or n-gram in the vocabulary.
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
    # Return the mapping of words to frequencies.
    return vocab


def __make_sentence(
    sentence: str,
    n: int,
    tokenize_pattern: re.Pattern = DELIMITERS,
    clean_pattern: re.Pattern = PUNCTUATION,
    stop_words: Set[str] = STOP_WORDS,
    drop_pattern: re.Pattern = STOP_PATTERN
) -> List[str]:
    '''
    Preprocess the sentence by tokenizing, cleaning, dropping stop words, and
    including any padding.

    sentence (str): collection of words to process.
    n (int): length of n-gram, i.e. number of words to represent with one token.
    tokenize_pattern (re.Pattern): regular expression for delimiters.
    clean_pattern (re.Pattern): regular expression for substrings to remove.
    stop_words (set): words to remove, i.e. articles, prepositions.
    drop_pattern (re.Pattern): regular expression for words to remove.

    Return processed sentence (list).
    '''
    # Tokenize the sentence on the delimiter.
    sentence = __tokenize_words(sentence, tokenize_pattern)
    # Clean words in the sentence.
    sentence = [__clean_words(word, clean_pattern) for word in sentence]
    # Drop any stop words in the sentence.
    sentence = __drop_words(sentence, stop_words, drop_pattern)
    # Pad both sides of the sentence.
    sentence = [PAD] * (n - 1) + sentence + [PAD] * (n - 1)
    # Return the sentence.
    return sentence


def __tokenize_words(text: str, pattern: re.Pattern) -> List[str]:
    '''
    Tokenize text on the delimiter.

    text (str): collection of characters to tokenize.
    pattern (re.Pattern): regular expression for delimiters.

    Return tokenized text (list).
    '''
    return re.split(pattern, text)


def __clean_words(text: str, pattern: re.Pattern) -> str:
    '''
    Clean text of substrings.

    text (str): collection of characters to clean.
    pattern (re.Pattern): regular expression for substrings to remove.

    Return cleaned text (str).
    '''
    return re.sub(pattern, '', text.lower())


def __drop_words(
    words: List[str],
    stop_words: Set[str],
    pattern: re.Pattern
) -> List[str]:
    '''
    Remove words that are among the stop words or match a regular expression.

    words (list): collection of words, i.e. a sentence.
    stop_words (set): words to remove.
    pattern (re.Pattern): regular expression for words to remove.

    Return words (list).
    '''
    return [
        word for word in words
        if word not in stop_words and not re.search(pattern, word)
    ]


def __make_index(vocab: Dict[str, int], k: int) -> Dict[str, int]:
    '''
    Collect indices for the k most frequent words in the vocabulary. Include
    tokens for unknown words and padding.

    vocab (dict): mapping of words to frequencies.
    k (int): number of most frequent words to keep.

    Return mapping of most frequent words to indices (dict).
    '''
    # Collect most the frequent words into a set.
    vocab = {word for word, _ in Counter(vocab).most_common(k)}
    # Include tokens for unknown words and padding.
    vocab.update([UNK, PAD])
    # Return the mapping of words to indices.
    return {word: i for i, word in enumerate(vocab)}


def __make_numeric(
    vocab: Dict[str, int],
    sentence: str,
    n: int,
    **kwargs: dict
) -> List[int]:
    '''
    Represent each word in the sentence with its index in the vocabulary.

    vocab (dict): mapping of words to indices.
    sentence (str): collection of words to process.
    n (int): length of n-gram, i.e. number of words to represent with one token.
    kwargs (dict): keyword arguments to pass to __make_sentence().

    Return the numericized sentence (list).
    '''
    new_sentence = []
    # Preprocess the sentence, i.e. tokenize, clean, drop, pad.
    sentence = __make_sentence(sentence, n, **kwargs)
    # Consider each word, or possibly n-gram, in the sentence.
    for w in range(len(sentence) - n):
        # Collect this word or n-gram.
        word = ' '.join(sentence[w:w + n])
        # Add the index of this word or n-gram to the new sentence.
        new_sentence.append(vocab.get(word, vocab[UNK]))
    # Return the numericized sentence.
    return new_sentence


def __make_vector(
    vocab: Dict[str, int],
    sentence: str,
    n: int,
    cbow: bool,
    **kwargs: dict
) -> torch.Tensor:
    '''
    Map words in a sentence to their indices in the vocabulary into either a
    simple numericization or a continuous bag of words.

    vocab (dict): mapping of words to indices.
    sentence (str): collection of words to process.
    n (int): length of n-gram, i.e. number of words to represent with one token.
    cbow (bool): whether to represent the sentence as a continuous bag of words.
    kwargs (dict): keyword arguments to pass to __make_sentence().

    Return the vectorization (torch.Tensor)
    '''
    # Numericize the sentence.
    sentence = __make_numeric(vocab, sentence, n, **kwargs)
    # Return the continuous bag of words.
    if cbow:
        # Count each word in the vocabulary that appears in the sentence.
        bag = torch.zeros(len(vocab))
        for idx in sentence:
            bag[idx] += 1
        return bag
    # Return the simple numericization otherwise.
    return torch.as_tensor(sentence)


def __make_batch(
    sentences: List[torch.Tensor],
    idx_padding: int,
    dtype: type
) -> torch.Tensor:
    '''
    Collect sentences into a batch with the length of the longest sentence.

    sentences (list): vectorized sentences sorted by length, descending.
    idx_padding (int): default value that pads shorter sentences.
    dtype (type): type of the values in the batch.

    Return the batch (torch.Tensor)
    '''
    # Initialize the batch; sentences are sorted, so the first is the longest.
    p = sentences[0].shape[0]
    q = len(sentences)
    batch = torch.full(size=(p, q), fill_value=idx_padding, dtype=dtype)
    # Load the batch with each sentence; unfilled cells have the padding value.
    for i, sentence in enumerate(sentences):
        batch[:sentence.shape[0], i] = sentence
    # Return the batch of sentences.
    return batch


def __show_label_stats(labels: np.ndarray) -> None:
    n = labels.shape[0]
    baseline = labels[labels == 1].shape[0] / n * 100
    print('Size of these data: %d' % n)
    print('Baseline precision: %7.4f' % baseline)


def __show_vocab_stats(vocab: Dict[str, int]) -> None:
    n = len(vocab)
    print('Size of the vocabulary: %d' % n)


def __show_batch_stats(dataset_batches: List[List[TorchTextLike]]) -> None:
    labels = ['training', 'validation', 'testing']
    for i, label in enumerate(labels):
        n = sum(batch.text.shape[1] for batch in dataset_batches[i])
        print('Size of the %s data: %d' % (label, n))
