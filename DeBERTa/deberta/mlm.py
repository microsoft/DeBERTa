# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This piece of code is modified based on https://github.com/huggingface/transformers

import torch
from torch import nn
import pdb

from .bert import LayerNorm,ACT2FN

__all__ = ['MLMPredictionHead']

class MLMPredictionHead(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, self.embedding_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

        self.LayerNorm = LayerNorm(self.embedding_size, config.layer_norm_eps)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.pre_norm = PreLayerNorm(config)

    def forward(self, hidden_states, embeding_weight):
        hidden_states = self.pre_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        # b x s x d
        hidden_states = MaskedLayerNorm(self.LayerNorm, hidden_states)

        # b x s x v
        logits = torch.matmul(hidden_states, embeding_weight.t().to(hidden_states)) + self.bias
        return logits
