#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import os
import codecs
import random
import logging
import json
import torch
import collections
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from .embedding_layer import EmbeddingLayer
from .elmo_model import ElmobiLm
from .token_model import ConvTokenEmbedder

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')

def read_list(sents, max_chars=None):
    dataset = []
    textset = []
    for sent in sents:
        data = ['<bos>']
        text = []
        for token in sent:
            text.append(token)
            if max_chars is not None and len(token) + 2 > max_chars:
                token = token[:max_chars - 2]
            data.append(token)
        data.append('<eos>')
        dataset.append(data)
        textset.append(text)
    return dataset, textset


def recover(li, ind):
    dummy = list(range(len(ind)))
    dummy.sort(key=lambda l: ind[l])
    li = [li[i] for i in dummy]
    return li


def create_one_batch(x, word2id, char2id, config, oov='<oov>', pad='<pad>', sort=True):
    batch_size = len(x)
    lst = list(range(batch_size))
    if sort:
        lst.sort(key=lambda l: -len(x[l]))
    x = [x[i] for i in lst]
    lens = [len(x[i]) for i in lst]
    max_len = max(lens)
    if word2id is not None:
        oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
        batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id)
        for i, x_i in enumerate(x):
            for j, x_ij in enumerate(x_i):
                batch_w[i][j] = word2id.get(x_ij, oov_id)
    else:
        batch_w = None
    if char2id is not None:
        bow_id, eow_id, oov_id, pad_id = [char2id.get(key, None) for key in ('<eow>', '<bow>', oov, pad)]

        if config['token_embedder']['name'].lower() == 'cnn':
            max_chars = config['token_embedder']['max_characters_per_token']
        elif config['token_embedder']['name'].lower() == 'lstm':
            max_chars = max([len(w) for i in lst for w in x[i]]) + 2
        else:
            raise ValueError('Unknown token_embedder: {0}'.format(config['token_embedder']['name']))

        batch_c = torch.LongTensor(batch_size, max_len, max_chars).fill_(pad_id)

        for i, x_i in enumerate(x):
            for j, x_ij in enumerate(x_i):
                batch_c[i][j][0] = bow_id
                if x_ij == '<bos>' or x_ij == '<eos>':
                    batch_c[i][j][1] = char2id.get(x_ij)
                    batch_c[i][j][2] = eow_id
                else:
                    for k, c in enumerate(x_ij):
                        batch_c[i][j][k + 1] = char2id.get(c, oov_id)
                    batch_c[i][j][len(x_ij) + 1] = eow_id
    else:
        batch_c = None

    masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]

    for i, x_i in enumerate(x):
        for j in range(len(x_i)):
            masks[0][i][j] = 1
            if j + 1 < len(x_i):
                masks[1].append(i * max_len + j)
            if j > 0:
                masks[2].append(i * max_len + j)

    masks[1] = torch.LongTensor(masks[1])
    masks[2] = torch.LongTensor(masks[2])

    return batch_w, batch_c, lens, masks


def create_batches(x, batch_size, word2id, char2id, config, perm=None, shuffle=False, sort=True, text=None):
    ind = list(range(len(x)))
    lst = perm or list(range(len(x)))
    if shuffle:
        random.shuffle(lst)

    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]
    ind = [ind[i] for i in lst]
    if text is not None:
        text = [text[i] for i in lst]

    sum_len = 0.0
    batches_w, batches_c, batches_lens, batches_masks, batches_text, batches_ind = [], [], [], [], [], []
    size = batch_size
    nbatch = (len(x) - 1) // size + 1
    for i in range(nbatch):
        start_id, end_id = i * size, (i + 1) * size
        bw, bc, blens, bmasks = create_one_batch(x[start_id: end_id], word2id, char2id, config, sort=sort)
        sum_len += sum(blens)
        batches_w.append(bw)
        batches_c.append(bc)
        batches_lens.append(blens)
        batches_masks.append(bmasks)
        batches_ind.append(ind[start_id: end_id])
        if text is not None:
            batches_text.append(text[start_id: end_id])

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_w = [batches_w[i] for i in perm]
        batches_c = [batches_c[i] for i in perm]
        batches_lens = [batches_lens[i] for i in perm]
        batches_masks = [batches_masks[i] for i in perm]
        batches_ind = [batches_ind[i] for i in perm]
        if text is not None:
            batches_text = [batches_text[i] for i in perm]

    recover_ind = [item for sublist in batches_ind for item in sublist]
    if text is not None:
        return batches_w, batches_c, batches_lens, batches_masks, batches_text, recover_ind
    return batches_w, batches_c, batches_lens, batches_masks, recover_ind


class Model(nn.Module):
  def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda=False):
    super(Model, self).__init__()
    self.use_cuda = use_cuda
    self.config = config

    self.token_embedder = ConvTokenEmbedder(
        config, word_emb_layer, char_emb_layer, use_cuda)

    self.encoder = ElmobiLm(config, use_cuda)

    self.output_dim = config['encoder']['projection_dim']

  def forward(self, word_inp, chars_package, mask_package):
    token_embedding = self.token_embedder(word_inp, chars_package, (mask_package[0].size(0), mask_package[0].size(1)))
    mask = Variable(mask_package[0]).cuda() if self.use_cuda else Variable(mask_package[0])
    encoder_output = self.encoder(token_embedding, mask)
    sz = encoder_output.size()
    token_embedding = torch.cat(
        [token_embedding, token_embedding], dim=2).view(1, sz[1], sz[2], sz[3])
    encoder_output = torch.cat(
        [token_embedding, encoder_output], dim=0)
    
    return encoder_output

  def load_model(self, path):
    self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl'),
                                                   map_location=lambda storage, loc: storage))
    self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl'),
                                            map_location=lambda storage, loc: storage))

def dict2namedtuple(dic):
  return collections.namedtuple('Namespace', dic.keys())(**dic)


class Embedder(object):
    def __init__(self, model_dir, batch_size=64):
        self.model_dir = model_dir
        self.model, self.config = self.get_model()
        self.batch_size = batch_size

    def get_model(self):
        torch.cuda.set_device(0)
        self.use_cuda = torch.cuda.is_available()
        args2 = dict2namedtuple(json.load(codecs.open(
            os.path.join(self.model_dir, 'config.json'), 'r', encoding='utf-8')))
        print(args2.config_path)
        with open(os.path.join(args2.config_path), 'r') as fin:
            config = json.load(fin)
        if config['token_embedder']['char_dim'] > 0:
            self.char_lexicon = {}
            with codecs.open(os.path.join(self.model_dir, 'char.dic'), 'r', encoding='utf-8') as fpi:
                for line in fpi:
                    tokens = line.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, i = tokens
                    self.char_lexicon[token] = int(i)
            char_emb_layer = EmbeddingLayer(
                config['token_embedder']['char_dim'], self.char_lexicon, fix_emb=False, embs=None)
            logging.info('char embedding size: ' +
                        str(len(char_emb_layer.word2id)))
        else:
            self.char_lexicon = None
            char_emb_layer = None

        if config['token_embedder']['word_dim'] > 0:
            self.word_lexicon = {}
            with codecs.open(os.path.join(self.model_dir, 'word.dic'), 'r', encoding='utf-8') as fpi:
                for line in fpi:
                    tokens = line.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, i = tokens
                    self.word_lexicon[token] = int(i)
            word_emb_layer = EmbeddingLayer(
                config['token_embedder']['word_dim'], self.word_lexicon, fix_emb=False, embs=None)
            logging.info('word embedding size: ' +
                        str(len(word_emb_layer.word2id)))
        else:
            self.word_lexicon = None
            word_emb_layer = None

        model = Model(config, word_emb_layer, char_emb_layer, self.use_cuda)

        if self.use_cuda:
            model.cuda()

        logging.info(str(model))
        model.load_model(self.model_dir)

        model.eval()
        return model, config

    def sents2elmo(self, sents, output_layer=-1):
        read_function = read_list

        test, text = read_function(sents, self.config['token_embedder']['max_characters_per_token'])

        test_w, test_c, test_lens, test_masks, test_text, recover_ind = create_batches(
            test, self.batch_size, self.word_lexicon, self.char_lexicon, self.config, text=text)

        cnt = 0

        self.model.eval()
        after_elmo = []
        for w, c, lens, masks, texts in zip(test_w, test_c, test_lens, test_masks, test_text):
            with torch.no_grad():
                output = self.model.forward(w, c, masks)
            for i, text in enumerate(texts):
                data = output[:, i, 1:lens[i]-1, :].data
                if self.use_cuda:
                    data = data.cpu()
                data = data.numpy()

                if output_layer == -1:
                    payload = np.average(data, axis=0)
                elif output_layer == -2:
                    payload = data
                else:
                    payload = data[output_layer]
                after_elmo.append(payload)

                cnt += 1
                if cnt % 1000 == 0:
                    logging.info('Finished {0} sentences.'.format(cnt))

        after_elmo = recover(after_elmo, recover_ind)
        return after_elmo
