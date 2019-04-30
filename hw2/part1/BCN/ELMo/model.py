#!/usr/bin/env python
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from .elmo_model import ElmobiLm
from .token_model import ConvTokenEmbedder


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
