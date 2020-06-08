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

def download_asset(name, tag=None, no_cache=False, cache_dir=None):
  _tag = tag
  if _tag is None:
    _tag = 'latest'
  if not cache_dir:
    cache_dir = os.path.join(pathlib.Path.home(), f'.~DeBERTa/assets/{_tag}/')
  os.makedirs(cache_dir, exist_ok=True)
  output=os.path.join(cache_dir, name)
  if os.path.exists(output) and (not no_cache):
    return output

  repo = 'https://api.github.com/repos/microsoft/DeBERTa/releases'
  releases = requests.get(repo).json()
  if tag and tag != 'latest':
    release = [r for r in releases if r['name'].lower()==tag.lower()]
    if len(release)!=1:
      raise Exception(f'{tag} can\'t be found in the repository.')
  else:
    release = releases[0]
  asset = [s for s in release['assets'] if s['name'].lower()==name.lower()]
  if len(asset)!=1:
    raise Exception(f'{name} can\'t be found in the release.')
  url = asset[0]['url']
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

def load_model_state(name, tag=None, no_cache=False, cache_dir=None):
  model_path = name
  if model_path and (not os.path.exists(model_path)) and not (('/' in model_path) or ('\\' in model_path)):
    _tag = tag
    if _tag is None:
      _tag = 'latest'
    if not cache_dir:
      cache_dir = os.path.join(pathlib.Path.home(), f'.~DeBERTa/assets/{_tag}/')
    os.makedirs(cache_dir, exist_ok=True)
    out_dir = os.path.join(cache_dir, name)
    model_path = os.path.join(out_dir, 'pytorch.model.bin')
    if (not os.path.exists(model_path)) or no_cache:
      asset = download_asset(name + '.zip', tag=tag, no_cache=no_cache, cache_dir=cache_dir)
      with ZipFile(asset, 'r') as zipf:
        for zip_info in zipf.infolist():
          if zip_info.filename[-1] == '/':
            continue
          zip_info.filename = os.path.basename(zip_info.filename)
          zipf.extract(zip_info, out_dir)
  elif not model_path:
    return None,None

  config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
  model_state = torch.load(model_path, map_location='cpu')
  logger.info("Loaded pre-trained model file {}".format(model_path))
  if 'config' in model_state:
    model_config = ModelConfig.from_dict(model_state['config'])
  elif os.path.exists(config_path):
    model_config = ModelConfig.from_json_file(config_path)
  return model_state, model_config

def load_vocab(name=None, tag=None, no_cache=False, cache_dir=None):
  if name is None:
    name = 'bpe_encoder'

  model_path = name
  if model_path and (not os.path.exists(model_path)) and not (('/' in model_path) or ('\\' in model_path)):
    _tag = tag
    if _tag is None:
      _tag = 'latest'
    if not cache_dir:
      cache_dir = os.path.join(pathlib.Path.home(), f'.~DeBERTa/assets/{_tag}/')
    os.makedirs(cache_dir, exist_ok=True)
    out_dir = os.path.join(cache_dir, name)
    model_path =os.path.join(out_dir, 'bpe_encoder.bin')
    if (not os.path.exists(model_path)) or no_cache:
      asset = download_asset(name + '.zip', tag=tag, no_cache=no_cache, cache_dir=cache_dir)
      with ZipFile(asset, 'r') as zipf:
        for zip_info in zipf.infolist():
          if zip_info.filename[-1] == '/':
            continue
          zip_info.filename = os.path.basename(zip_info.filename)
          zipf.extract(zip_info, out_dir)
  elif not model_path:
    return None,None

  encoder_state = torch.load(model_path)
  return encoder_state

def test_download():
  vocab = load_vocab()
