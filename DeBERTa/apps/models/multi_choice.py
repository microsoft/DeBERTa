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
from torch.nn import CrossEntropyLoss
import math

from ...deberta import *
from ...utils import *

__all__ = ['MultiChoiceModel']
class MultiChoiceModel(NNModule):
  def __init__(self, config, num_labels = 2, drop_out=None, **kwargs):
    super().__init__(config)
    self.num_labels = num_labels
    self._register_load_state_dict_pre_hook(self._pre_load_hook)
    self.deberta = DeBERTa(config)
    self.config = config
    pool_config = PoolConfig(self.config)
    output_dim = self.deberta.config.hidden_size
    self.pooler = ContextPooler(pool_config)
    output_dim = self.pooler.output_dim()
    drop_out = config.hidden_dropout_prob if drop_out is None else drop_out
    self.classifier = torch.nn.Linear(output_dim, 1)
    self.dropout = StableDropout(drop_out)
    self.apply(self.init_weights)
    self.deberta.apply_state()

  def forward(self, input_ids, type_ids=None, input_mask=None, labels=None, position_ids=None, **kwargs):
    num_opts = input_ids.size(1)
    input_ids = input_ids.view([-1, input_ids.size(-1)])
    if type_ids is not None:
      type_ids = type_ids.view([-1, type_ids.size(-1)])
    if position_ids is not None:
      position_ids = position_ids.view([-1, position_ids.size(-1)])
    if input_mask is not None:
      input_mask = input_mask.view([-1, input_mask.size(-1)])
    outputs = self.deberta(input_ids, token_type_ids=type_ids, attention_mask=input_mask,
        position_ids=position_ids, output_all_encoded_layers=True)
    hidden_states = outputs['hidden_states'][-1]
    logits = self.classifier(self.dropout(self.pooler(hidden_states)))
    logits = logits.float().squeeze(-1)
    logits = logits.view([-1, num_opts])
    loss = 0
    if labels is not None:
      labels = labels.long()
      loss_fn = CrossEntropyLoss()
      loss = loss_fn(logits, labels)

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
