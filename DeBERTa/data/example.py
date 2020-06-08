import torch
import os
from collections import OrderedDict
import numpy as np
import tempfile
import numpy as np
import mmap
import pickle
import signal
import sys
import pdb

from ..utils import xtqdm as tqdm

__all__=['ExampleInstance', 'example_to_feature', 'ExampleSet']

class ExampleInstance:
  def __init__(self, segments, label=None,  **kwv):
    self.segments = segments
    self.label = label
    self.__dict__.update(kwv)

  def __repr__(self):
    return f'segments: {self.segments}\nlabel: {self.label}'

  def __getitem__(self, i):
    return self.segments[i]

  def __len__(self):
    return len(self.segments)

class ExampleSet:
  def __init__(self, pairs):
    self._data = np.array([pickle.dumps(p) for p in pairs])
    self.total = len(self._data)

  def __getitem__(self, idx):
    """
    return pair
    """
    if isinstance(idx, tuple):
      idx,rng, ext_params = idx
    else:
      rng,ext_params=None, None
    content = self._data[idx]
    example = pickle.loads(content)
    return example

  def __len__(self):
    return self.total

  def __iter__(self):
    for i in range(self.total):
      yield self[i]

def _truncate_segments(segments, max_num_tokens, rng):
  """
  Truncate sequence pair according to original BERT implementation:
  https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L391
  """
  while True:
    if sum(len(s) for s in segments)<=max_num_tokens:
      break

    segments = sorted(segments, key=lambda s:len(s), reverse=True)
    trunc_tokens = segments[0]

    assert len(trunc_tokens) >= 1

    if rng.random() < 0.5:
      trunc_tokens.pop(0)
    else:
      trunc_tokens.pop()
  return segments

def example_to_feature(tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, label_type='int', **kwargs):
  if not rng:
    rng = random
  max_num_tokens = max_seq_len - len(example.segments) - 1
  segments = _truncate_segments([tokenizer.tokenize(s) for s in example.segments], max_num_tokens, rng)
  tokens = ['[CLS]']
  type_ids = [0]
  for i,s in enumerate(segments):
    tokens.extend(s)
    tokens.append('[SEP]')
    type_ids.extend([i]*(len(s)+1))
  if mask_generator:
    tokens, lm_labels = mask_generator.mask_tokens(tokens, rng)
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  pos_ids = list(range(len(token_ids)))
  input_mask = [1]*len(token_ids)
  features = OrderedDict(input_ids = token_ids,
      type_ids = type_ids,
      position_ids = pos_ids,
      input_mask = input_mask)
  if mask_generator:
    features['lm_labels'] = lm_labels
  padding_size = max(0, max_seq_len - len(token_ids))
  for f in features:
    features[f].extend([0]*padding_size)
    features[f] = torch.tensor(features[f], dtype=torch.int)
  label_type = torch.int if label_type=='int' else torch.float
  if example.label is not None:
    features['labels'] = torch.tensor(example.label, dtype=label_type)
  return features
