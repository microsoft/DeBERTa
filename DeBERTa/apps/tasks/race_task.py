#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#

from glob import glob
from collections import OrderedDict,defaultdict,Sequence
import copy
import math
from scipy.special import softmax
import numpy as np
import pdb
import os
import sys
import csv

import random
import torch
import re
import ujson as json
from .metrics import *
from .task import EvalData, Task
from .task_registry import register_task
from ...utils import xtqdm as tqdm
from ...data import ExampleInstance, ExampleSet, DynamicDataset,example_to_feature
from ...data.example import *
from ...utils import get_logger
from ..models.multi_choice import MultiChoiceModel

logger=get_logger()

__all__ = ["RACETask"]

@register_task(name="RACE", desc="ReAding Comprehension dataset collected from English Examinations, http://www.qizhexie.com/data/RACE_leaderboard.html")
class RACETask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    middle = self.load_jsonl(os.path.join(self.data_dir, 'train_middle.jsonl'))
    high = self.load_jsonl(os.path.join(self.data_dir, 'train_high.jsonl'))
    examples = ExampleSet(middle + high)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('test-high', 'test_high.jsonl', 'test', ignore_metric=True),
        self._data('test-middle', 'test_middle.jsonl', 'test', ignore_metric=True),
        self._data('test', ['test_middle.jsonl', 'test_high.jsonl'], 'test'),
        ]
   
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('high', 'test_high.jsonl', 'test'),
        self._data('middle', 'test_middle.jsonl', 'test'),
        ]
    
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev', ignore_metric=False):
    if isinstance(path, str):
      path = [path]
    data = []
    for p in path:
      input_src = os.path.join(self.data_dir, p)
      assert os.path.exists(input_src), f"{input_src} doesn't exists"
      data.extend(self.load_jsonl(input_src))

    predict_fn = self.get_predict_fn()
    examples = ExampleSet(data)
    return EvalData(name, examples,
      metrics_fn = self.get_metrics_fn(), predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy'])

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      metrics =  OrderedDict(accuracy= metric_accuracy(logits, labels))
      return metrics
    return metrics_fn
  
  def get_labels(self):
    """See base class."""
    return ["A", "B", "C", "D"]

  def load_jsonl(self, path):
    examples = []
    with open(path, encoding='utf-8') as fs:
      data = [json.loads(l) for l in fs]
      for d in data:
        page = d["article"]
        for q,o,a in zip(d["questions"], d["options"], d["answers"]):
          example = ExampleInstance(segments=[page, q, *o], label=self.label2id(a))
          examples.append(example)
    return examples

  def get_feature_fn(self, max_seq_len = 512, mask_gen = None):
    def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
      return self.example_to_feature(self.tokenizer, example, max_seq_len = max_seq_len, \
        rng = rng, mask_generator = mask_gen, ext_params = ext_params, **kwargs)
    return _example_to_feature

  def example_to_feature(self, tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, label_type='int', **kwargs):
    if not rng:
      rng = random
    max_num_tokens = max_seq_len - 3
    def _normalize(text):
      text = re.sub(r'\s+', ' ', text.strip('\t \r\n_').replace('\n', ' ')).strip()
      return text

    # page,question,options
    context = tokenizer.tokenize(_normalize(example.segments[0]))
    features = OrderedDict(input_ids = [],
        type_ids = [],
        position_ids = [],
        input_mask = [])
    for option in example.segments[2:]:
      #TODO: truncate
      question = example.segments[1]
      qa_cat = " ".join([question, option])
      qa_cat = tokenizer.tokenize(_normalize(qa_cat))[:160]

      segments = [context, qa_cat]
      segments = _truncate_segments(segments, max_num_tokens, rng)
      tokens = ['[CLS]']
      type_ids = [0]
      for i,s in enumerate(segments):
        tokens.extend(s)
        tokens.append('[SEP]')
        type_ids.extend([i]*(len(s)+1))
      token_ids = tokenizer.convert_tokens_to_ids(tokens)
      pos_ids = list(range(len(token_ids)))
      rel_pos = []
      input_mask = [1]*len(token_ids)
      features['input_ids'].append(token_ids)
      features['type_ids'].append(type_ids)
      features['position_ids'].append(pos_ids)
      features['input_mask'].append(input_mask)
      padding_size = max(0, max_seq_len - len(token_ids))
      for f in features:
        features[f][-1].extend([0]*padding_size)

    for f in features:
      features[f] = torch.tensor(features[f], dtype=torch.int)
    if example.label is not None:
      label_type = torch.int if label_type=='int' else torch.float
      features['labels'] = torch.tensor(example.label, dtype=label_type)
    return features

  def get_model_class_fn(self):
    def partial_class(*wargs, **kwargs):
      return MultiChoiceModel.load_model(*wargs, **kwargs)
    return partial_class
