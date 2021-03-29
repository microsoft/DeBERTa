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

import math
import torch
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from ...deberta import *
from ...utils import *

__all__ = ['ReCoRDQAModel']

class ReCoRDQAModel(NNModule):
  def __init__(self, config, drop_out=None, **kwargs):
    super().__init__(config)
    self.deberta = DeBERTa(config)
    self.config = config
    self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
    self.classifier = torch.nn.Linear(config.hidden_size, 1)
    drop_out = config.hidden_dropout_prob if drop_out is None else drop_out
    self.dropout = StableDropout(drop_out)
    self.apply(self.init_weights)
    self.deberta.apply_state()

  def forward(self, input_ids, entity_indice, type_ids=None, input_mask=None, labels=None, position_ids=None, placeholder=None, **kwargs):
    outputs = self.deberta(input_ids, attention_mask=input_mask, token_type_ids=type_ids,\
        position_ids=position_ids, output_all_encoded_layers=True)
    encoder_layers = outputs['hidden_states']
    # bxexsp
    entity_mask = entity_indice>0
    tokens = encoder_layers[-1]
    # bxexspxd
    entities = torch.gather(tokens.unsqueeze(1).expand(entity_indice.size()[:2]+tokens.size()[1:]), index=entity_indice.long().unsqueeze(-1).expand(entity_indice.size()+(tokens.size(-1),)), dim=-2)
    ctx = tokens[:,:1,:]/math.sqrt(tokens.size(-1))
    # bxsx1
    att_score = torch.matmul(tokens, ctx.transpose(-1,-2))
    # bxexspx1
    entity_score = torch.gather(att_score.unsqueeze(1).expand(entity_indice.size()[:2]+att_score.size()[1:]), index=entity_indice.long().unsqueeze(-1).expand(entity_indice.size()+(att_score.size(-1),)), dim=-2)
    entity_score = entity_score.squeeze(-1)*entity_mask.to(entity_score) - (1-entity_mask.to(entity_score))*10000.0
    att_prob = torch.nn.functional.softmax(entity_score, dim=-1).unsqueeze(-2)
    # bxexd
    entity_ebd = torch.matmul(att_prob, entities).squeeze(-2)

    entity_ebd = self.proj(entity_ebd)
    entity_ebd = ACT2FN['gelu'](entity_ebd)

    sequence_out = self.dropout(entity_ebd)
    logits = self.classifier(sequence_out).float().squeeze(-1)
    entity_mask = (entity_mask.sum(-1)>0).to(logits)
    logits = logits*entity_mask + (entity_mask-1)*10000.0
    loss = 0
    if labels is not None:
      entity_index = entity_mask.view(-1).nonzero().view(-1)
      sp_logits = logits.view(-1)
      labels = labels.view(-1)
      sp_logits = torch.gather(sp_logits, index=entity_index, dim=0)
      labels = torch.gather(labels, index=entity_index, dim=0)
      loss_fn = BCEWithLogitsLoss()
      loss = loss_fn(sp_logits, labels.to(sp_logits))

    return {
        'logits': logits,
        'loss': loss }
