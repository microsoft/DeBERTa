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

__all__ = ['NNModule']

class NNModule(nn.Module):
  """ An abstract class to handle weights initialization and \
    a simple interface for dowloading and loading pretrained models.

  Args:
    
    config (:obj:`~DeBERTa.deberta.ModelConfig`): The model config to the module

  """

  def __init__(self, config, *inputs, **kwargs):
    super().__init__()
    self.config = config

  def init_weights(self, module):
    """ Apply Gaussian(mean=0, std=`config.initializer_range`) initialization to the module.

    Args:
      
      module (:obj:`torch.nn.Module`): The module to apply the initialization.
    
    Example::
      
      class MyModule(NNModule):
        def __init__(self, config):
          # Add construction instructions
          self.bert = DeBERTa(config)
          
          # Add other modules
          ...

          # Apply initialization
          self.apply(self.init_weights)

    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @classmethod
  def load_model(cls, model_path, model_config=None, tag=None, no_cache=False, cache_dir=None , *inputs, **kwargs):
    """ Instantiate a sub-class of NNModule from a pre-trained model file.
      
    Args:

      model_path (:obj:`str`): Path or name of the pre-trained model which can be either,
        
        - The path of pre-trained model

        - The pre-trained DeBERTa model name in `DeBERTa GitHub releases <https://github.com/microsoft/DeBERTa/releases>`_, i.e. [**base, base_mnli, large, large_mnli**].

        If `model_path` is `None` or `-`, then the method will create a new sub-class without initialing from pre-trained models.

      model_config (:obj:`str`): The path of model config file. If it's `None`, then the method will try to find the the config in order:
        
        1. ['config'] in the model state dictionary.

        2. `model_config.json` aside the `model_path`.
        
        If it failed to find a config the method will fail.

      tag (:obj:`str`, optional): The release tag of DeBERTa, default: `None`.

      no_cache (:obj:`bool`, optional): Disable local cache of downloaded models, default: `False`.

      cache_dir (:obj:`str`, optional): The cache directory used to save the downloaded models, default: `None`. If it's `None`, then the models will be saved at `$HOME/.~DeBERTa`

    Return:
      
      :obj:`NNModule` : The sub-class object.

    """
    # Load config
    if model_config:
      config = ModelConfig.from_json_file(model_config)
    else:
      config = None
    model_config = None
    model_state = None
    if (model_path is not None) and (model_path.strip() == '-' or model_path.strip()==''):
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
          'max_position_embeddings'] or (k not in  model_config.__dict__) or (model_config.__dict__[k] < 0):
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
    logger.warning(f'Missing keys: {missing_keys}, unexpected_keys: {unexpected_keys}, error_msgs: {error_msgs}')
    return model
