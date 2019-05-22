import numpy as np
import tensorflow as tf
from text_utils import transform_texts
import pickle
import csv
import pandas as pd
import logging

def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

class Logger(object):
    def __init__(self, path):
        logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')
        self._logger = logging.getLogger('trainlogger')
        self._logger.info('Train-Logger started ...')

    def log(self, **kwargs):
        self._logger.info(kwargs)

def get_paragraphs():
    paragraphs = pd.read_csv('./Data/prediction_train.tsv', sep='\t', encoding='latin1')
    p = paragraphs['paragraph_text_without_last_sentence'] + paragraphs['paragraph_last_sentence']
    return list(p.dropna())
    

def encode_dataset():
    paragraphs = get_paragraphs()
    tokens, masks = transform_texts(paragraphs)
    with open('Data/tokens.pkl', 'wb') as pkl:
        pickle.dump(tokens, pkl)
    with open('Data/masks.pkl', 'wb') as pkl:
        pickle.dump(masks, pkl)


def get_validation():
    with open('Data/tokens.pkl', 'rb') as pkl:
        tokens = pickle.load(pkl)
    with open('Data/masks.pkl', 'rb') as pkl:
        masks = pickle.load(pkl)

    n = len(tokens) // 10
    return tokens[-n:], masks[-n:]

def iter_data(n_batch, n_epochs = None, train = True):
    with open('Data/tokens.pkl', 'rb') as pkl:
        tokens = pickle.load(pkl)
    with open('Data/masks.pkl', 'rb') as pkl:
        masks = pickle.load(pkl)

    if train:
        n = len(tokens) - 5000
        for epoch in range(n_epochs):
            pi = np.random.permutation(n)
            tokens = tokens[pi]
            masks = masks[pi]

            for i in range(0, n, n_batch):
                if i + n_batch > n:
                    break
                yield (tokens[i:i + n_batch], masks[i:i + n_batch])

    else:
        n = 5000
        tokens, masks = tokens[-n:], masks[-n:]
        pi = np.random.permutation(n)
        tokens, masks = tokens[pi], masks[pi]

        for i in range(0, n, n_batch):
            if i + n_batch > n:
                break
            yield (tokens[i:i + n_batch], masks[i:i + n_batch])
