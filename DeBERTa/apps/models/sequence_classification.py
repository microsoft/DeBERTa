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

import pdb
import torch
from torch.nn import CrossEntropyLoss
import math

from ...deberta import *
from ...utils import *

__all__= ['SequenceClassificationModel']
class SequenceClassificationModel(NNModule):
  def __init__(self, config, num_labels=2, drop_out=None, pre_trained=None):
    super().__init__(config)
    self.num_labels = num_labels
    self._register_load_state_dict_pre_hook(self._pre_load_hook)
    self.deberta = DeBERTa(config, pre_trained=pre_trained)
    if pre_trained is not None:
      self.config = self.deberta.config
    else:
      self.config = config
    pool_config = PoolConfig(self.config)
    output_dim = self.deberta.config.hidden_size
    self.pooler = ContextPooler(pool_config)
    output_dim = self.pooler.output_dim()

    self.classifier = torch.nn.Linear(output_dim, num_labels)
    drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
    self.dropout = StableDropout(drop_out)
    self.apply(self.init_weights)
    self.deberta.apply_state()

  def forward(self, input_ids, type_ids=None, input_mask=None, labels=None, position_ids=None, **kwargs):
    outputs = self.deberta(input_ids, attention_mask=input_mask, token_type_ids=type_ids,
        position_ids=position_ids, output_all_encoded_layers=True)
    encoder_layers = outputs['hidden_states']
    pooled_output = self.pooler(encoder_layers[-1])
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    loss = 0
    if labels is not None:
      if self.num_labels ==1:
        # regression task
        loss_fn = torch.nn.MSELoss()
        logits=logits.view(-1).to(labels.dtype)
        loss = loss_fn(logits, labels.view(-1))
      elif labels.dim()==1 or labels.size(-1)==1:
        label_index = (labels >= 0).nonzero()
        labels = labels.long()
        if label_index.size(0) > 0:
          labeled_logits = torch.gather(logits, 0, label_index.expand(label_index.size(0), logits.size(1)))
          labels = torch.gather(labels, 0, label_index.view(-1))
          loss_fct = CrossEntropyLoss()
          loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
        else:
          loss = torch.tensor(0).to(logits)
      else:
        log_softmax = torch.nn.LogSoftmax(-1)
        label_confidence = 1
        loss = -((log_softmax(logits)*labels).sum(-1)*label_confidence).mean()

    return {
            'logits' : logits,
            'loss' : loss
          }
    return (logits,loss)

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
