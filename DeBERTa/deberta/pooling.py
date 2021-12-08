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
import pdb
from .bert import ACT2FN
from .ops import StableDropout
from .config import AbsModelConfig

__all__ = ['PoolConfig', 'ContextPooler']

class PoolConfig(AbsModelConfig):
    """Configuration class to store the configuration of `pool layer`.

        Parameters:
        
            config (:class:`~DeBERTa.deberta.ModelConfig`): The model config. The field of pool config will be initalized with the `pooling` field in model config.

        Attributes:

            hidden_size (int): Size of the encoder layers and the pooler layer, default: `768`.

            dropout (float): The dropout rate applied on the output of `[CLS]` token,

            hidden_act (:obj:`str`): The activation function of the projection layer, it can be one of ['gelu', 'tanh'].

        Example::

            # Here is the content of an exmple model config file in json format

                {
                  "hidden_size": 768,
                  "num_hidden_layers" 12,
                  "num_attention_heads": 12,
                  "intermediate_size": 3072,
                  ...
                  "pooling": {
                    "hidden_size":  768,
                    "hidden_act": "gelu",
                    "dropout": 0.1
                  }
                }

    """
    def __init__(self, config=None):
        """Constructs PoolConfig.

        Args:
           `config`: the config of the model. The field of pool config will be initalized with the 'pooling' field in model config.
        """
        
        self.hidden_size = 768
        self.dropout = 0
        self.hidden_act = 'gelu'
        if config:
            pool_config = getattr(config, 'pooling', config)
            if isinstance(pool_config, dict):
                pool_config = AbsModelConfig.from_dict(pool_config)
            self.hidden_size = getattr(pool_config, 'hidden_size', config.hidden_size)
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
