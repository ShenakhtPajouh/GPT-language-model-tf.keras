import numpy as np
import re
import ftfy
import json
import spacy
from nltk import sent_tokenize
from tqdm import tqdm

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
        
    return pairs
    
def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub('\s*\n\s*', ' \n ', text)
    text = re.sub('[^\S\n]+', ' ', text)
    return text.strip()
    
class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v:k for k,v in self.encoder.items()}
        merges = open(bpe_path).read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def bpe(self, token):
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def __call__(self, texts, verbose = True):
        texts_tokens = []
        
        if verbose:
            for text in tqdm(texts, ncols = 80, leave = False):
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
                
        else:
            for text in texts:
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
                
        return texts_tokens


def transform_texts(texts, n_ctx=512, n_vocab=40478, encoder=None, target = 'LM'):
    if encoder == None:
        ENCODER_PATH = 'model/encoder_bpe_40000.json'
        BPE_PATH = 'model/vocab_40000.bpe'
        encoder = TextEncoder(ENCODER_PATH, BPE_PATH)

    tokens = encoder(texts, verbose=False)
    n_batch = len(tokens)
    inputs = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    masks = np.zeros((n_batch, n_ctx), dtype=np.float32)
    
    if target == 'LM':
        for i, x in enumerate(tokens):
            if i % 1000 == 0:
                print(i)
            token = x[:n_ctx]
            inputs[i, :len(token), 0] = token
            masks[i, :len(token)] = 1
            
    else:
        j = 0
        for i, x in enumerate(tokens):
            sents = sent_tokenize(texts[i])
            if len(sents) <= 1:
                continue

            x1 = x[:n_ctx]
            last_sent_len = len(encoder([sents[-1]], verbose = False)[0])
            l1 = len(x1) - last_sent_len
            inputs[j, :len(x1), 0] = x1
            masks[j, l1:len(x1)] = 1
            j += 1


    inputs[:, :, 1] = np.arange(n_vocab, n_vocab + n_ctx)
    return inputs, masks
