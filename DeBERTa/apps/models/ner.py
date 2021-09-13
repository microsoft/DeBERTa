# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import math
from torch import nn
from torch.nn import CrossEntropyLoss
from ...deberta import DeBERTa,NNModule,ACT2FN,StableDropout

__all__ = ['NERModel']

class NERModel(NNModule):
  def __init__(self, config, num_labels = 2, drop_out=None, **kwargs):
    super().__init__(config)
    self._register_load_state_dict_pre_hook(self._pre_load_hook)
    self.deberta = DeBERTa(config)
    self.num_labels = num_labels
    self.proj = nn.Linear(config.hidden_size, config.hidden_size)
    self.classifier = nn.Linear(config.hidden_size, self.num_labels)
    drop_out = config.hidden_dropout_prob if drop_out is None else drop_out
    self.dropout = StableDropout(drop_out)
    self.apply(self.init_weights)

  def forward(self, input_ids, type_ids=None, input_mask=None, labels=None, position_ids=None, **kwargs):
    outputs = self.deberta(input_ids, token_type_ids=type_ids, attention_mask=input_mask, \
        position_ids=position_ids, output_all_encoded_layers=True)
    encoder_layers = outputs['hidden_states']
    cls = encoder_layers[-1]
    cls = self.proj(cls)
    cls = ACT2FN['gelu'](cls)
    cls = self.dropout(cls)
    logits = self.classifier(cls).float()
    loss = 0
    if labels is not None:
      labels = labels.long().view(-1)
      label_index = (labels>=0).nonzero().view(-1)
      valid_labels = labels.index_select(dim=0, index=label_index)
      valid_logits = logits.view(-1, logits.size(-1)).index_select(dim=0, index=label_index)
      loss_fn = CrossEntropyLoss()
      loss = loss_fn(valid_logits, valid_labels)

    return {
            'logits' : logits,
            'loss' : loss
          }

  def _pre_load_hook(self, state_dict, prefix, local_metadata, strict,
      missing_keys, unexpected_keys, error_msgs):
    new_state = dict()
    bert_prefix = prefix + 'bert.'
    deberta_prefix = prefix + 'deberta.'
    for k in list(state_dict.keys()):
      if k.startswith(bert_prefix):
        nk = deberta_prefix + k[len(bert_prefix):]
        value = state_dict[k]
        del state_dict[k]
        state_dict[nk] = value
