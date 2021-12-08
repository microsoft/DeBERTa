# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This piece of code is modified based on https://github.com/huggingface/transformers

import copy
import torch
from torch import nn
from collections import Sequence
from packaging import version
import numpy as np
import math
import os
import pdb

import json
from .ops import *
from .disentangled_attention import *
from .da_utils import *

__all__ = ['BertEncoder', 'BertEmbeddings', 'ACT2FN', 'LayerNorm', 'BertLMPredictionHead']

class BertSelfOutput(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
    self.dropout = StableDropout(config.hidden_dropout_prob)
    self.config = config

  def forward(self, hidden_states, input_states, mask=None):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states += input_states
    hidden_states = MaskedLayerNorm(self.LayerNorm, hidden_states)
    return hidden_states

class BertAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.self = DisentangledSelfAttention(config)
    self.output = BertSelfOutput(config)
    self.config = config

  def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
    output = self.self(hidden_states, attention_mask, return_att, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
    self_output, att_matrix, att_logits_=output['hidden_states'], output['attention_probs'], output['attention_logits']
    if query_states is None:
      query_states = hidden_states
    attention_output = self.output(self_output, query_states, attention_mask)

    if return_att:
      return (attention_output, att_matrix)
    else:
      return attention_output

class BertIntermediate(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.intermediate_act_fn = ACT2FN[config.hidden_act] \
      if isinstance(config.hidden_act, str) else config.hidden_act

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states

class BertOutput(nn.Module):
  def __init__(self, config):
    super(BertOutput, self).__init__()
    self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
    self.dropout = StableDropout(config.hidden_dropout_prob)
    self.config = config

  def forward(self, hidden_states, input_states, mask=None):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states += input_states
    hidden_states = MaskedLayerNorm(self.LayerNorm, hidden_states)
    return hidden_states

class BertLayer(nn.Module):
  def __init__(self, config):
    super(BertLayer, self).__init__()
    self.attention = BertAttention(config)
    self.intermediate = BertIntermediate(config)
    self.output = BertOutput(config)

  def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
    attention_output = self.attention(hidden_states, attention_mask, return_att=return_att, \
      query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
    if return_att:
      attention_output, att_matrix = attention_output
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output, attention_mask)
    if return_att:
      return (layer_output, att_matrix)
    else:
      return layer_output

class ConvLayer(nn.Module):
    def __init__(self, config):
      super().__init__()
      kernel_size = getattr(config, 'conv_kernel_size', 3)
      groups = getattr(config, 'conv_groups', 1)
      self.conv_act = getattr(config, 'conv_act', 'tanh')
      self.conv = torch.nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size, padding = (kernel_size-1)//2, groups = groups)
      self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
      self.dropout = StableDropout(config.hidden_dropout_prob)
      self.config = config

    def forward(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
        if version.Version(torch.__version__) >= version.Version('1.2.0a'):
            rmask = (1-input_mask).bool()
        else:
            rmask = (1-input_mask).byte()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = ACT2FN[self.conv_act](self.dropout(out))
        output_states = MaskedLayerNorm(self.LayerNorm, residual_states + out, input_mask)

        return output_states

class BertEncoder(nn.Module):
  """ Modified BertEncoder with relative position bias support
  """
  def __init__(self, config):
    super().__init__()
    #layer = BertLayer(config)
    self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    self.relative_attention = getattr(config, 'relative_attention', False)
    if self.relative_attention:
      self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
      if self.max_relative_positions <1:
        self.max_relative_positions = config.max_position_embeddings
      self.position_buckets = getattr(config, 'position_buckets', -1)
      pos_ebd_size = self.max_relative_positions*2
      if self.position_buckets>0:
        pos_ebd_size = self.position_buckets*2
      self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

    self.norm_rel_ebd = [x.strip() for x in getattr(config, 'norm_rel_ebd', 'none').lower().split('|')]
    if 'layer_norm' in self.norm_rel_ebd:
      self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine = True)
    kernel_size = getattr(config, 'conv_kernel_size', 0)
    self.with_conv = False
    if kernel_size > 0:
      self.with_conv = True
      self.conv = ConvLayer(config)

  def get_rel_embedding(self):
    rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
    if rel_embeddings is not None and ('layer_norm' in self.norm_rel_ebd):
      rel_embeddings = self.LayerNorm(rel_embeddings)
    return rel_embeddings

  def get_attention_mask(self, attention_mask):
    if attention_mask.dim()<=2:
      extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
      attention_mask = extended_attention_mask*extended_attention_mask.squeeze(-2).unsqueeze(-1)
      attention_mask = attention_mask.byte()
    elif attention_mask.dim()==3:
      attention_mask = attention_mask.unsqueeze(1)

    return attention_mask

  def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
    if self.relative_attention and relative_pos is None:
      q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
      relative_pos = build_relative_position(q, hidden_states.size(-2), bucket_size = self.position_buckets, max_position=self.max_relative_positions)
    return relative_pos

  def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, return_att=False, query_states = None, relative_pos=None):
    if attention_mask.dim()<=2:
      input_mask = attention_mask
    else:
      input_mask = (attention_mask.sum(-2)>0).byte()
    attention_mask = self.get_attention_mask(attention_mask)
    relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

    all_encoder_layers = []
    att_matrices = []
    if isinstance(hidden_states, Sequence):
      next_kv = hidden_states[0]
    else:
      next_kv = hidden_states
    rel_embeddings = self.get_rel_embedding()
    for i, layer_module in enumerate(self.layer):
      output_states = layer_module(next_kv, attention_mask, return_att, query_states = query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
      if return_att:
        output_states, att_m = output_states

      if i == 0 and self.with_conv:
        prenorm = output_states #output['prenorm_states']
        output_states = self.conv(hidden_states, prenorm, input_mask)

      if query_states is not None:
        query_states = output_states
        if isinstance(hidden_states, Sequence):
          next_kv = hidden_states[i+1] if i+1 < len(self.layer) else None
      else:
        next_kv = output_states

      if output_all_encoded_layers:
        all_encoder_layers.append(output_states)
        if return_att:
          att_matrices.append(att_m)
    if not output_all_encoded_layers:
      all_encoder_layers.append(output_states)
      if return_att:
        att_matrices.append(att_m)
    return {
        'hidden_states': all_encoder_layers,
        'attention_matrices': att_matrices
        }

class BertEmbeddings(nn.Module):
  """Construct the embeddings from word, position and token_type embeddings.
  """
  def __init__(self, config):
    super(BertEmbeddings, self).__init__()
    padding_idx = getattr(config, 'padding_idx', 0)
    self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)
    self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx = padding_idx)
    self.position_biased_input = getattr(config, 'position_biased_input', True)
    self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

    if config.type_vocab_size>0:
      self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)
    
    if self.embedding_size != config.hidden_size:
      self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
    self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
    self.dropout = StableDropout(config.hidden_dropout_prob)
    self.output_to_half = False
    self.config = config

  def forward(self, input_ids, token_type_ids=None, position_ids=None, mask = None):
    seq_length = input_ids.size(1)
    if position_ids is None:
      position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
      position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    words_embeddings = self.word_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids.long())

    embeddings = words_embeddings
    if self.config.type_vocab_size>0:
      token_type_embeddings = self.token_type_embeddings(token_type_ids)
      embeddings += token_type_embeddings

    if self.position_biased_input:
      embeddings += position_embeddings

    if self.embedding_size != self.config.hidden_size:
      embeddings = self.embed_proj(embeddings)
    embeddings = MaskedLayerNorm(self.LayerNorm, embeddings, mask)
    embeddings = self.dropout(embeddings)
    return {
        'embeddings': embeddings,
        'position_embeddings': position_embeddings}

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, self.embedding_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

        self.LayerNorm = LayerNorm(self.embedding_size, config.layer_norm_eps, elementwise_affine=True)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states, embeding_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        # b x s x d
        hidden_states = MaskedLayerNorm(self.LayerNorm, hidden_states)

        # b x s x v
        logits = torch.matmul(hidden_states, embeding_weight.t().to(hidden_states)) + self.bias
        return logits
