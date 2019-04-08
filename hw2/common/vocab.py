from collections import namedtuple

import numpy as np
from tqdm import tqdm


SpecialToken = namedtuple('SpecialToken', ['sym', 'idx'])


class SpecialVocab:
    def __init__(self, special_tokens):
        self._special_tokens = special_tokens
        for i, tok in enumerate(special_tokens):
            setattr(self, tok, SpecialToken(sym='<{}>'.format(tok), idx=i))

    def __len__(self):
        return len(self._special_tokens)

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self):
        if self._iter_idx < len(self):
            self._iter_idx += 1
            return getattr(self, self._special_tokens[self._iter_idx - 1])
        raise StopIteration


def load_embedding(embedding_path):
    with open(embedding_path) as f:
        lines = f.readlines()
    if len(lines[0].strip().split()) == 2:
        lines = lines[1:]
    emb = {}
    bar = tqdm(
        lines, desc='[*] Loading embedding from {}'.format(embedding_path),
        dynamic_ncols=True)
    for l in bar:
        if '\xa0' in l or '\x85' in l:
            continue
        v, *e = l.strip().split(' ')
        emb[v.lower()] = list(map(float, e))
    bar.close()

    return emb


class Vocab:
    def __init__(self, tokens, special_tokens, embedding_path=None,
                 freeze_embedding=None, embedding_dimension=None, **kwargs):
        self._special = SpecialVocab(special_tokens)
        self._iv = [v.sym for v in self._special] + tokens
        self._vi = {v: i for i, v in enumerate(self._iv)}
        if embedding_path:
            if freeze_embedding is None:
                raise ValueError('Vocab: Please specify whether the embedding should be'
                                 'freezed or not')
            self.freeze_emb = freeze_embedding
            emb = load_embedding(embedding_path)
            self._emb_dim = len(emb['the'])
            self._ie = np.random.normal(
                size=(len(self._special) + len(tokens), self._emb_dim))
            self._ie[self._special.pad.idx] = np.zeros(self._emb_dim)
            for i, t in enumerate(tokens):
                idx = len(self._special) + i
                if t in emb:
                    self._ie[idx] = np.array(emb[t])
        else:
            if freeze_embedding is not None:
                raise ValueError('Vocab: No need to specify freeze_embedding when '
                                 'embedding_path is not provided')
            self._emb_dim = embedding_dimension
            self._ie = None

    def vtoi(self, v):
        return self._vi.get(v, self._special.unk.idx)

    def itov(self, i):
        return self._iv[i]

    @property
    def emb_dim(self):
        return self._emb_dim

    @property
    def emb(self):
        return self._ie

    @property
    def sp(self):
        return self._special

    @property
    def n_sp(self):
        return len(self._special)

    def __len__(self):
        return len(self._vi)
