#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#

from glob import glob
from collections import OrderedDict,defaultdict
from collections.abc import Sequence
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
from DeBERTa.apps.tasks.metrics import *
from DeBERTa.apps.tasks import EvalData, Task,register_task
from DeBERTa.utils import xtqdm as tqdm
from DeBERTa.data import ExampleInstance, ExampleSet, DynamicDataset,example_to_feature
from DeBERTa.data.example import *
from DeBERTa.utils import get_logger
from DeBERTa.data.example import _truncate_segments
from DeBERTa.apps.models.multi_choice import MultiChoiceModel

logger=get_logger()

__all__ = ["AlphaNLITask"]

@register_task(name="AlphaNLI", desc="Abductive Natural Language Inference(aNLI). https://leaderboard.allenai.org/anli/submissions/public")
class AlphaNLITask(Task):
  def __init__(self, data_dir, tokenizer, **kwargs):
    super().__init__(tokenizer, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    train = self.load_data(os.path.join(self.data_dir, 'train'))
    examples = ExampleSet(train)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', 'dev', 'dev'),
        self._data('train', 'train', 'dev', max_examples=1000, shuffle=True),
        ]
   
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test', 'test')
        ]
    
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev', ignore_metric=False, max_examples=None, shuffle=False):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src + '.jsonl'), f"{input_src} doesn't exists"
    data = self.load_data(input_src, max_examples=max_examples, shuffle=shuffle)
    examples = ExampleSet(data)
    predict_fn = self.get_predict_fn()
    metrics_fn = self.get_metrics_fn()
    return EvalData(name, examples,
      metrics_fn = metrics_fn, predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy'])

  def get_predict_fn(self):
    def predict_fn(logits, output_dir, name, prefix):
      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      probs = softmax(logits, -1)[:,1]
      probs = np.reshape(probs, (len(logits)//2, 2))
      preds = np.argmax(probs, axis=-1)
      labels = ["1", "2"]
      with open(output, 'w', encoding='utf-8') as fs:
        for i,p in enumerate(preds):
          fs.write(labels[p] + '\n')

    return predict_fn

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metric_accuracy(logits, labels):
      predicts = np.argmax(logits, axis=1)
      return accuracy_score(labels, predicts)

    def metrics_fn(logits, labels):
      return OrderedDict(accuracy = metric_multi_accuracy(logits, labels, 2))
    return metrics_fn

  def get_feature_fn(self, max_seq_len = 512, mask_gen = None):
    def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
      return self.example_to_feature(self.tokenizer, example, max_seq_len = max_seq_len, \
        rng = rng, mask_generator = mask_gen, ext_params = ext_params, **kwargs)
    return _example_to_feature

  def get_labels(self):
    """See base class."""
    return [0, 1]

  def load_data(self, path, max_seq_len=512, max_examples=None, shuffle=False):
    with open(path + '.jsonl') as fs:
      data = [json.loads(l) for l in fs]

    if os.path.exists(path + '-labels.lst'):
      with open(path + '-labels.lst') as fs:
        labels = [int(l.strip())-1 for l in fs]
    else:
      labels = [None] * len(data)

    for  d,l in zip(data, labels):
      d['label'] = l

    examples=[]
    sub_data = data
    if max_examples is not None and max_examples>0:
      if shuffle:
        random.shuffle(sub_data)
        sub_data = sub_data[:max_examples]

    for d in sub_data:
      passage = self.tokenizer.tokenize(d['obs1'] + ' ' + d['obs2'])
      label = d['label']
      for i,k in enumerate([d["hyp1"], d["hyp2"]]):
        l = None if label is None else i==label
        segments = [passage]
        opt = self.tokenizer.tokenize(k)
        segments.append(opt)
        examples.append(ExampleInstance(segments, label=l))

    def get_stats(l):
      return f'Max={max(l)}, min={min(l)}, avg={np.mean(l)}'
    ctx_token_size = [len(e.segments[0]) for e in examples]
    opt1_token_size = [len(e.segments[1]) for e in examples]
    total_size = [len(s) + len(e.segments[0]) for e in examples for s in e.segments[1:]]
    logger.info(f'Premise statistics: {get_stats(ctx_token_size)}, long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}')
    #logger.info(f'Question statistics: {get_stats(q_token_size)}')
    logger.info(f'Opt1 statistics: {get_stats(opt1_token_size)}')
    logger.info(f'Total statistics: {get_stats(total_size)}, long={len([t for t in total_size if t>500])}')

    return examples

  def example_to_feature(self, tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, label_type='int', **kwargs):
    if not rng:
      rng = random
    def make_pair_feature(segments):
      max_num_tokens = max_seq_len - len(segments) - 1
      features = OrderedDict()
      tokens = ['[CLS]']
      type_ids = [0]
      segments = _truncate_segments(segments, max_num_tokens, rng)
      for i,s in enumerate(segments):
        tokens.extend(s)
        tokens.append('[SEP]')
        type_ids.extend([i]*(len(s)+1))
      token_ids = tokenizer.convert_tokens_to_ids(tokens)
      pos_ids = list(range(len(token_ids)))
      input_mask = [1]*len(token_ids)
      features['input_ids'] = token_ids
      features['type_ids'] = type_ids
      features['position_ids'] = pos_ids
      features['input_mask'] = input_mask
      padding_size = max(0, max_seq_len - len(token_ids))
      for f in features:
        features[f].extend([0]*padding_size)
      return features
    features = make_pair_feature([example.segments[0], example.segments[1]])
    for f in features:
      features[f] = torch.tensor(features[f], dtype=torch.int)

    if example.label is not None: # and example.label[0]>=0 and example.label[1]>=0:
      label_type = torch.int if label_type=='int' else torch.float
      features['labels'] = torch.tensor(example.label, dtype=label_type)
    return features
