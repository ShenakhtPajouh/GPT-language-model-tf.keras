import numpy as np
import tensorflow as tf
from text_utils import transform_texts, TextEncoder
import pickle
import math
import pandas as pd
import logging

def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def stop_gradients(target, mask):
    mask_h = tf.abs(mask-1)
    return tf.stop_gradient(mask_h * target) + mask * target

class Logger(object):
    def __init__(self, path):
        logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')
        self._logger = logging.getLogger('trainlogger')
        self._logger.info('Train-Logger started ...')

    def log(self, **kwargs):
        # print(kwargs)
        self._logger.info(kwargs)

def get_paragraphs():
    paragraphs = pd.read_csv('./Data/prediction_train.tsv', sep='\t', encoding='latin1')
    p =  paragraphs['paragraph_text_without_last_sentence'] + paragraphs['paragraph_last_sentence']
    return list(p.dropna())

def encode_dataset():
    tokens, masks = transform_texts(get_paragraphs())
    with open('Data/tokens.pkl', 'wb') as pkl:
        pickle.dump(tokens, pkl)
    with open('Data/masks.pkl', 'wb') as pkl:
        pickle.dump(masks, pkl)

def encode(encoder=None):
    if encoder == None:
        ENCODER_PATH = 'model/encoder_bpe_40000.json'
        BPE_PATH = 'model/vocab_40000.bpe'
        encoder = TextEncoder(ENCODER_PATH, BPE_PATH)

    tokens = encoder(get_paragraphs(), verbose=False)
    with open('Data/tokens.pkl', 'wb') as pkl:
        pickle.dump(tokens, pkl)


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
        n = len(tokens) - (len(tokens) // 10)
        for epoch in range(n_epochs):
            pi = np.random.permutation(n)
            tokens = tokens[pi]
            masks = masks[pi]

            for i in range(0, n, n_batch):
                if i + n_batch > n:
                    break
                yield (tokens[i:i + n_batch], masks[i:i + n_batch])

    else:
        n = len(tokens) // 10
        tokens, masks = tokens[-n:], masks[-n:]
        pi = np.random.permutation(n)
        tokens, masks = tokens[pi], masks[pi]

        for i in range(0, n, n_batch):
            if i + n_batch > n:
                break
            yield (tokens[i:i + n_batch], masks[i:i + n_batch])

def gelu(x):
    """
    Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.
    Returns:
      `input_tensor` with the GELU activation applied.
    """
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))


def swish(x):
    """
    Swish tends to work better than ReLU on deeper models across a number of challenging data sets.
    For further information:
    medium.com/@neuralnets/swish-activation-function-by-google-53e1ea86f820

    Args:
      input_tensor: float Tensor to perform activation.
    Returns:
      `input_tensor` with the swish activation applied.
    """
    return x * tf.nn.sigmoid(x)


def dropout(input_tensor, dropout_prob, train):
    """
      Perform dropout.
      Args:
        input_tensor: inpout tensor.
        dropout_prob: the probability of dropping out a value

      Returns:
        A version of `input_tensor` with dropout applied.
    """
    if not train or dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output
