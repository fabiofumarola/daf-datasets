

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ...utils.data_utils import get_file

import numpy as np
import json

import nltk
from nltk.corpus import stopwords
import string


nltk.download('stopwords')
nltk.download('punkt')

pad_char = 0
start_char = 1
oov_char = 2
index_from = 3


def get_stopwords():
    punctuation = [c for c in string.punctuation]
    punctuation += [',', '.', '-',
                    '"', "'", ':',
                    ';', '(', ')',
                    '[', ']', '{', '}',
                    '’', '”', '“', '``', "''", '–']
    stop_words = set(stopwords.words('italian'))
    stop_words.update(punctuation)
    return stop_words


def load_data(path='data_dirigenti.npz', num_words=None, skip_top=0,
              remove_stopwords=False, seed=11235):
    """Loads the atti-dirigenti dataset.
    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).
        num_words: max number of words to include. Words are ranked
            by how often they occur (in the training set) and only
            the most frequent words are kept
        skip_top: skip the top N most frequently occurring words
            (which may not be informative).
        seed: random seed for sample shuffling.
        start_char: The start of a sequence will be marked with this character.
            Set to 1 because 0 is usually the padding character.
        oov_char: words that were cut out because of the `num_words`
            or `skip_top` limit will be replaced with this character.
        index_from: index actual words with this index and higher.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    # Raises

    Note that the 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `num_words` cut here.
    Words that were not seen in the training set but are in the test set
    have simply been skipped.
    """
    path = get_file(path,
                    origin='https://github.com/fabiofumarola/daf-datasets/raw/master/data/data_dirigenti.npz',
                    file_hash='c981a46aeb147cc22eb7b151abdb479f')

    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    # shuffle the indices for train and test
    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    x_all = np.concatenate([x_train, x_test])
    labels_all = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        x_all = [[start_char] + [w + index_from for w in sequence
                                 ] for sequence in x_all]
    elif index_from:
        x_all = [[w + index_from for w in sequence] for sequence in x_all]

    if remove_stopwords:
        # get the stopwords from the word index
        word_index = get_word_index()
        stopwords = get_stopwords()
        # get the index of the stopwords
        index_stopwords = {word_index[s] for s in stopwords if s in stopwords}
        x_all = [[w for w in sequence if w not in index_stopwords
                  ] for sequence in x_all]

    if not num_words:
        num_words = max([max(x) for x in x_all])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        x_all = [[w if skip_top <= w < num_words else oov_char for w in seq
                  ] for seq in x_all]
    else:
        x_all = [[w for w in seq if skip_top <= w < num_words
                  ] for seq in x_all]

    idx = len(x_train)
    x_train, y_train = np.array(x_all[:idx]), np.array(labels_all[:idx])
    x_test, y_test = np.array(x_all[idx:]), np.array(labels_all[:idx])

    return (x_train, y_train), (x_test, y_test)


def get_word_index(path='data_dirigenti_word_index.json'):
    """Retrieves the dictionary mapping word indices back to words.

    # Arguments
        path: where to cache the data (relative to `~/.daf/dataset`).

    # Returns
        The word index dictionary.
    """
    path = get_file(path,
                    origin='https://github.com/fabiofumarola/daf-datasets/raw/master/data/data_dirigenti_word_index.json',
                    file_hash=None)
    with open(path, 'r') as f:
        return json.load(f)


def get_label_index(path='data_dirigenti_label_index.json'):
    """Retrieves the dictionary mapping labels indices back to words.

    # Arguments
        path: where to cache the data (relative to `~/.daf/dataset`).

    # Returns
        The word index dictionary.
    """
    path = get_file(path,
                    origin='https://github.com/fabiofumarola/daf-datasets/raw/master/data/data_dirigenti_label_index.json',
                    file_hash=None)
    with open(path, 'r') as f:
        return json.load(f)
