from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from typing import Callable
from overrides import overrides


class Highway(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1,
                 activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu) -> None:
        super(Highway, self).__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 2)
                                            for _ in range(num_layers)])
        self._activation = activation
        for layer in self._layers:
            layer.bias[input_dim:].data.fill_(1)

    @overrides
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part = projected_input[:, (0 * self._input_dim):(1 * self._input_dim)]
            gate = projected_input[:, (1 * self._input_dim):(2 * self._input_dim)]
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class ConvTokenEmbedder(nn.Module):
  def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda):
    super(ConvTokenEmbedder, self).__init__()
    self.config = config
    self.use_cuda = use_cuda

    self.word_emb_layer = word_emb_layer
    self.char_emb_layer = char_emb_layer

    self.output_dim = config['encoder']['projection_dim']
    self.emb_dim = 0
    if word_emb_layer is not None:
      self.emb_dim += word_emb_layer.n_d

    if char_emb_layer is not None:
      self.convolutions = []
      cnn_config = config['token_embedder']
      filters = cnn_config['filters']
      char_embed_dim = cnn_config['char_dim']

      for i, (width, num) in enumerate(filters):
        conv = torch.nn.Conv1d(
          in_channels=char_embed_dim,
          out_channels=num,
          kernel_size=width,
          bias=True
        )
        self.convolutions.append(conv)

      self.convolutions = nn.ModuleList(self.convolutions)
      
      self.n_filters = sum(f[1] for f in filters)
      self.n_highway = cnn_config['n_highway']

      self.highways = Highway(self.n_filters, self.n_highway, activation=torch.nn.functional.relu)
      self.emb_dim += self.n_filters

    self.projection = nn.Linear(self.emb_dim, self.output_dim, bias=True)
    
  def forward(self, word_inp, chars_inp, shape):
    embs = []
    batch_size, seq_len = shape
    if self.word_emb_layer is not None:
      batch_size, seq_len = word_inp.size(0), word_inp.size(1)
      word_emb = self.word_emb_layer(Variable(word_inp).cuda() if self.use_cuda else Variable(word_inp))
      embs.append(word_emb)

    if self.char_emb_layer is not None:
      chars_inp = chars_inp.view(batch_size * seq_len, -1)

      character_embedding = self.char_emb_layer(Variable(chars_inp).cuda() if self.use_cuda else Variable(chars_inp))

      character_embedding = torch.transpose(character_embedding, 1, 2)

      cnn_config = self.config['token_embedder']
      if cnn_config['activation'] == 'tanh':
        activation = torch.nn.functional.tanh
      elif cnn_config['activation'] == 'relu':
        activation = torch.nn.functional.relu
      else:
        raise Exception("Unknown activation")

      convs = []
      for i in range(len(self.convolutions)):
        convolved = self.convolutions[i](character_embedding)
        # (batch_size * sequence_length, n_filters for this width)
        convolved, _ = torch.max(convolved, dim=-1)
        convolved = activation(convolved)
        convs.append(convolved)
      char_emb = torch.cat(convs, dim=-1)
      char_emb = self.highways(char_emb)

      embs.append(char_emb.view(batch_size, -1, self.n_filters))
      
    token_embedding = torch.cat(embs, dim=2)

    return self.projection(token_embedding)
