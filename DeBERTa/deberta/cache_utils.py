# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 05/15/2020
#

import pdb
import torch
import os
import requests
from .config import ModelConfig
import pathlib
from ..utils import xtqdm as tqdm
from zipfile import ZipFile
from ..utils import get_logger
logger = get_logger()

__all__ = ['pretrained_models', 'load_model_state', 'load_vocab']

class PretrainedModel:
  def __init__(self, name, vocab, vocab_type, model='pytorch_model.bin', config='config.json', **kwargs):
    self.__dict__.update(kwargs)
    host = f'https://huggingface.co/microsoft/{name}/resolve/main/'
    self.name = name
    self.model_url = host + model
    self.config_url = host + config
    self.vocab_url = host + vocab
    self.vocab_type = vocab_type
  
pretrained_models= {
    'base': PretrainedModel('deberta-base', 'bpe_encoder.bin', 'gpt2'),
    'large': PretrainedModel('deberta-large', 'bpe_encoder.bin', 'gpt2'),
    'xlarge': PretrainedModel('deberta-xlarge', 'bpe_encoder.bin', 'gpt2'),
    'base-mnli': PretrainedModel('deberta-base-mnli', 'bpe_encoder.bin', 'gpt2'),
    'large-mnli': PretrainedModel('deberta-large-mnli', 'bpe_encoder.bin', 'gpt2'),
    'xlarge-mnli': PretrainedModel('deberta-xlarge-mnli', 'bpe_encoder.bin', 'gpt2'),
    'xlarge-v2': PretrainedModel('deberta-v2-xlarge', 'spm.model', 'spm'),
    'xxlarge-v2': PretrainedModel('deberta-v2-xxlarge', 'spm.model', 'spm'),
    'xlarge-v2-mnli': PretrainedModel('deberta-v2-xlarge-mnli', 'spm.model', 'spm'),
    'xxlarge-v2-mnli': PretrainedModel('deberta-v2-xxlarge-mnli', 'spm.model', 'spm'),
    'deberta-v3-small': PretrainedModel('deberta-v3-small', 'spm.model', 'spm'),
    'deberta-v3-base': PretrainedModel('deberta-v3-base', 'spm.model', 'spm'),
    'deberta-v3-large': PretrainedModel('deberta-v3-large', 'spm.model', 'spm'),
    'mdeberta-v3-base': PretrainedModel('mdeberta-v3-base', 'spm.model', 'spm'),
    'deberta-v3-xsmall': PretrainedModel('deberta-v3-xsmall', 'spm.model', 'spm'),
  }

def download_asset(url, name, tag=None, no_cache=False, cache_dir=None):
  _tag = tag
  if _tag is None:
    _tag = 'latest'
  if not cache_dir:
    cache_dir = os.path.join(pathlib.Path.home(), f'.~DeBERTa/assets/{_tag}/')
  os.makedirs(cache_dir, exist_ok=True)
  output=os.path.join(cache_dir, name)
  if os.path.exists(output) and (not no_cache):
    return output

  #repo=f'https://huggingface.co/microsoft/deberta-{name}/blob/main/bpe_encoder.bin'
  headers = {}
  headers['Accept'] = 'application/octet-stream'
  resp = requests.get(url, stream=True, headers=headers)
  if resp.status_code != 200:
    raise Exception(f'Request for {url} return {resp.status_code}, {resp.text}')
  
  try:
    with open(output, 'wb') as fs:
      progress = tqdm(total=int(resp.headers['Content-Length']) if 'Content-Length' in resp.headers else -1, ncols=80, desc=f'Downloading {name}')
      for c in resp.iter_content(chunk_size=1024*1024):
        fs.write(c)
        progress.update(len(c))
      progress.close()
  except:
    os.remove(output)
    raise

  return output

def load_model_state(path_or_pretrained_id, tag=None, no_cache=False, cache_dir=None):
  model_path = path_or_pretrained_id
  if model_path and (not os.path.exists(model_path)) and (path_or_pretrained_id.lower() in pretrained_models):
    _tag = tag
    pretrained = pretrained_models[path_or_pretrained_id.lower()]
    if _tag is None:
      _tag = 'latest'
    if not cache_dir:
      cache_dir = os.path.join(pathlib.Path.home(), f'.~DeBERTa/assets/{_tag}/{pretrained.name}')
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, 'pytorch_model.bin')
    if (not os.path.exists(model_path)) or no_cache:
      asset = download_asset(pretrained.model_url, 'pytorch_model.bin', tag=tag, no_cache=no_cache, cache_dir=cache_dir)
      asset = download_asset(pretrained.config_url, 'model_config.json', tag=tag, no_cache=no_cache, cache_dir=cache_dir)
  elif not model_path:
    return None,None

  config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
  model_state = torch.load(model_path, map_location='cpu')
  logger.info("Loaded pretrained model file {}".format(model_path))
  if 'config' in model_state:
    model_config = ModelConfig.from_dict(model_state['config'])
  elif os.path.exists(config_path):
    model_config = ModelConfig.from_json_file(config_path)
  else:
    model_config = None
  return model_state, model_config

def load_vocab(vocab_path=None, vocab_type=None, pretrained_id=None, tag=None, no_cache=False, cache_dir=None):
  if pretrained_id and (pretrained_id.lower() in pretrained_models):
    _tag = tag
    if _tag is None:
      _tag = 'latest'

    pretrained = pretrained_models[pretrained_id.lower()]
    if not cache_dir:
      cache_dir = os.path.join(pathlib.Path.home(), f'.~DeBERTa/assets/{_tag}/{pretrained.name}')
    os.makedirs(cache_dir, exist_ok=True)
    vocab_type = pretrained.vocab_type
    url = pretrained.vocab_url
    outname = os.path.basename(url)
    vocab_path =os.path.join(cache_dir, outname)
    if (not os.path.exists(vocab_path)) or no_cache:
      asset = download_asset(url, outname, tag=tag, no_cache=no_cache, cache_dir=cache_dir)
  if vocab_type is None:
    vocab_type = 'spm'
  return vocab_path, vocab_type

def test_download():
  vocab = load_vocab()
