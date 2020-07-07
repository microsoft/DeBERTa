# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/15/2020
#

import copy
import torch
import os

import json
from .ops import *
from .bert import *
from .config import ModelConfig
from .cache_utils import load_model_state

__all__ = ['DeBERTa']

class DeBERTa(torch.nn.Module):
  """ DeBERTa encoder
  This module is composed of the input embedding layer with stacked transformer layers with disentangled attention.

  Parameters:
    config:
      A model config class instance with the configuration to build a new model. The schema is similar to `BertConfig`, \
          for more details, please refer :class:`~DeBERTa.deberta.ModelConfig`

    pre_trained:
      The pre-trained DeBERTa model, it can be a physical path of a pre-trained DeBERTa model or a released configurations, \
          i.e. [**base, large, base_mnli, large_mnli**]

  """

  def __init__(self, config=None, pre_trained=None):
    super().__init__()
    if config:
      self.z_steps = getattr(config, 'z_steps', 0)
    else:
      self.z_steps = 0

    state = None
    if pre_trained is not None:
      state, model_config = load_model_state(pre_trained)
      if config is not None and model_config is not None:
        for k in config.__dict__:
          if k not in ['hidden_size',
            'intermediate_size',
            'num_attention_heads',
            'num_hidden_layers',
            'vocab_size',
            'max_position_embeddings']:
            model_config.__dict__[k] = config.__dict__[k]
      config = copy.copy(model_config)
    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.config = config
    self.pre_trained = pre_trained
    self.apply_state(state)

  def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_all_encoded_layers=True, position_ids = None, return_att = False):
    """
    Args:
      input_ids:
        a torch.LongTensor of shape [batch_size, sequence_length] \
      with the word token indices in the vocabulary

      attention_mask:
        an optional parameter for input mask or attention mask.

        - If it's an input mask, then it will be torch.LongTensor of shape [batch_size, sequence_length] with indices \
      selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max \
      input sequence length in the current batch. It's the mask that we typically use for attention when \
      a batch has varying length sentences.

        - If it's an attention mask then it will be torch.LongTensor of shape [batch_size, sequence_length, sequence_length]. \
      In this case, it's a mask indicate which tokens in the sequence should be attended by other tokens in the sequence.

      token_type_ids:
        an optional torch.LongTensor of shape [batch_size, sequence_length] with the token \
      types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to \
      a `sentence B` token (see BERT paper for more details).

      output_all_encoded_layers:
        whether to output results of all encoder layers, default, True

    Returns:

      - The output of the stacked transformer layers if `output_all_encoded_layers=True`, else \
      the last layer of stacked transformer layers

      - Attention matrix of self-attention layers if `return_att=True`


    Example::

      # Batch of wordPiece token ids.
      # Each sample was padded with zero to the maxium length of the batch
      input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
      # Mask of valid input ids
      attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

      # DeBERTa model initialized with pretrained base model
      bert = DeBERTa(pre_trained='base')

      encoder_layers = bert(input_ids, attention_mask=attention_mask)

    """

    if attention_mask is None:
      attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    embedding_output = self.embeddings(input_ids.to(torch.long), token_type_ids.to(torch.long), position_ids, attention_mask)
    encoded_layers = self.encoder(embedding_output,
                   attention_mask,
                   output_all_encoded_layers=output_all_encoded_layers, return_att = return_att)
    if return_att:
      encoded_layers, att_matrixs = encoded_layers

    if self.z_steps>1:
      hidden_states = encoded_layers[-2]
      layers = [self.encoder.layer[-1] for _ in range(z_steps)]
      query_states = encoded_layers[-1]
      rel_embeddings = self.encoder.get_rel_embedding()
      attention_mask = self.encoder.get_attention_mask(attention_mask)
      rel_pos = self.encoder.get_rel_pos(embedding_output)
      for layer in layers[1:]:
        query_states = layer(hidden_states, attention_mask, return_att=False, query_states = query_states, relative_pos=rel_pos, rel_embeddings=rel_embeddings)
        encoded_layers.append(query_states)

    if not output_all_encoded_layers:
      encoded_layers = encoded_layers[-1:]

    if return_att:
      return encoded_layers, att_matrixs
    return encoded_layers

  def apply_state(self, state = None):
    """ Load state from previous loaded model state dictionary.

      Args:
        state (:obj:`dict`, optional): State dictionary as the state returned by torch.module.state_dict(), default: `None`. \
            If it's `None`, then will use the pre-trained state loaded via the constructor to re-initialize \
            the `DeBERTa` model
    """
    if self.pre_trained is None and state is None:
      return
    if state is None:
      state, config = load_model_state(self.pre_trained)
      self.config = config

    def key_match(key, s):
      c = [k for k in s if key in k]
      assert len(c)==1, c
      return c[0]
    current = self.state_dict()
    for c in current.keys():
      current[c] = state[key_match(c, state.keys())]
    self.load_state_dict(current)
