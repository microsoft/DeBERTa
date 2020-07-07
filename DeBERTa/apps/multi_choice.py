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

from ..deberta import *
from ..utils import *
import pdb

__all__ = ['MultiChoiceModel']
class MultiChoiceModel(NNModule):
  def __init__(self, config, num_labels = 2, drop_out=None, **kwargs):
    super().__init__(config)
    self.bert = DeBERTa(config)
    self.num_labels = num_labels
    self.classifier = nn.Linear(config.hidden_size, 1)
    drop_out = config.hidden_dropout_prob if drop_out is None else drop_out
    self.dropout = StableDropout(drop_out)
    self.apply(self.init_weights)

  def forward(self, input_ids, type_ids=None, input_mask=None, labels=None, position_ids=None, **kwargs):
    num_opts = input_ids.size(1)
    input_ids = input_ids.view([-1, input_ids.size(-1)])
    if type_ids is not None:
      type_ids = type_ids.view([-1, type_ids.size(-1)])
    if position_ids is not None:
      position_ids = position_ids.view([-1, position_ids.size(-1)])
    if input_mask is not None:
      input_mask = input_mask.view([-1, input_mask.size(-1)])
    encoder_layers = self.bert(input_ids, token_type_ids=type_ids, attention_mask=input_mask,
        position_ids=position_ids, output_all_encoded_layers=True)
    seqout = encoder_layers[-1]
    cls = seqout[:,:1,:]
    cls = cls/math.sqrt(seqout.size(-1))
    att_score = torch.matmul(cls, seqout.transpose(-1,-2))
    att_mask = input_mask.unsqueeze(1).to(att_score)
    att_score = att_mask*att_score + (att_mask-1)*10000.0
    att_score = torch.nn.functional.softmax(att_score, dim=-1)
    pool = torch.matmul(att_score, seqout).squeeze(-2)
    cls = self.dropout(pool)
    logits = self.classifier(cls).float().squeeze(-1)
    logits = logits.view([-1, num_opts])
    loss = 0
    if labels is not None:
      labels = labels.long()
      loss_fn = CrossEntropyLoss()
      loss = loss_fn(logits, labels)

    return (logits, loss)

