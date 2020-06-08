import pdb
import os
import torch
import copy
from torch import nn
from .config import ModelConfig
from ..utils import xtqdm as tqdm
from .cache_utils import load_model_state

from ..utils import get_logger
logger = get_logger()


class NNModule(nn.Module):
  """ An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
  """

  def __init__(self, config, *inputs, **kwargs):
    super().__init__()
    self.config = config

  def init_weights(self, module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @classmethod
  def load_model(cls, model_path, bert_config=None, tag=None, no_cache=False, cache_dir=None , *inputs, **kwargs):
    """
    Instantiate a NNModule from a pre-trained model file.
    """
    # Load config
    if bert_config:
      config = ModelConfig.from_json_file(bert_config)
    else:
      config = None
    model_config = None
    model_state = None
    if model_path.strip() == '-' or model_path.strip()=='':
      model_path = None
    try:
      model_state, model_config = load_model_state(model_path, tag=tag, no_cache=no_cache, cache_dir=cache_dir)
    except Exception as exp:
      raise Exception(f'Failed to get model {model_path}. Exception: {exp}')
    
    if config is not None and model_config is not None:
      for k in config.__dict__:
        if k not in ['hidden_size',
          'intermediate_size',
          'num_attention_heads',
          'num_hidden_layers',
          'vocab_size',
          'max_position_embeddings']:
          model_config.__dict__[k] = config.__dict__[k]
    if model_config is not None:
      config = copy.copy(model_config)
    vocab_size = config.vocab_size
    # Instantiate model.
    model = cls(config, *inputs, **kwargs)
    if not model_state:
      return model
    # copy state_dict so _load_from_state_dict can modify it
    state_dict = model_state.copy()

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    def load(module, prefix=''):
      local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
      module._load_from_state_dict(
        state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
      for name, child in module._modules.items():
        if child is not None:
          load(child, prefix + name + '.')
    load(model)
    return model
