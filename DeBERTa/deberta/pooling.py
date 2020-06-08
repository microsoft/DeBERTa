#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#
"""
Pooling functions
"""

from torch import nn
import copy
import json
from .bert import ACT2FN
from .ops import StableDropout

class PoolConfig(object):
    """Configuration class to store the configuration of `attention pool layer`.
    """
    def __init__(self, model_config):
        """Constructs PoolConfig.

        Params:
           `model_config`: the config of the model. The field of pool config will be initalized with the 'pooling' field in model config.
        """
        pool_config = getattr(model_config, 'pooling', model_config)
        self.hidden_size = getattr(pool_config, 'hidden_size', model_config.hidden_size)
        self.dropout = getattr(pool_config, 'dropout', 0)
        self.hidden_act = getattr(pool_config, 'hidden_act', 'gelu')

class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = StableDropout(config.dropout)
        self.config = config

    def forward(self, hidden_states, mask = None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.hidden_act](pooled_output)
        return pooled_output

    def output_dim(self):
        return self.config.hidden_size
