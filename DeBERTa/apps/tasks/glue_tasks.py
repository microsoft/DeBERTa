# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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
from ...utils import get_logger
from ...data import ExampleInstance, ExampleSet, DynamicDataset,example_to_feature
from ...data.example import _truncate_segments
from ...data.example import *

logger=get_logger()

__all__ = ["MNLITask", "ANLITask", "STSBTask", "SST2Task", "QQPTask", "ColaTask", "MRPCTask", "RTETask", "QNLITask"]

@register_task("sts-b")
class STSBTask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    input_src = os.path.join(self.data_dir, 'train.tsv')
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    examples = ExampleSet([ExampleInstance((l[7], l[8]), float(l[9])) for l in data[1:]]) # if l[3] in ['slate']])
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen, label_type='float'), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', 'dev.tsv', 'dev')
        ]
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, label_type='float'), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.tsv', 'test')
        ]
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, label_type='float'), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev'):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    predict_fn = self.get_predict_fn()
    if type_name=='test':
      examples = ExampleSet([ExampleInstance((l[7], l[8])) for l in data[1:]])
    else:
      examples = ExampleSet([ExampleInstance((l[7], l[8]), float(l[9])) for l in data[1:]])

    return EvalData(name, examples,
      metrics_fn = self.get_metrics_fn(), predict_fn = predict_fn)

  def get_metrics_fn(self):
    def metric_fn(logits, labels):
      return OrderedDict(
      pearsonr=pearsonr(labels, logits)[0],
      spearmanr= spearmanr(labels, logits)[0])
    return metric_fn

  def get_predict_fn(self):
    """Calcuate metrics based on prediction results"""
    def predict_fn(logits, output_dir, name, prefix):
      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('index\tpredictions\n')
        for i,p in enumerate(np.squeeze(logits)):
          fs.write('{}\t{}\n'.format(i, p))

    return predict_fn
  
  def get_labels(self):
    """See base class."""
    return ["1"]

@register_task("rte")
class RTETask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    input_src = os.path.join(self.data_dir, 'train.tsv')
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    examples = [ExampleInstance((l[1],l[2]), self.label2id(l[3])) for l in data[1:]] # if l[3] in ['slate']])

    examples = ExampleSet(examples)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, extra_data=None, **kwargs):
    ds = [
        self._data('dev', "dev.tsv", 'dev'),
        ]

    if extra_data is not None:
      extra_data = extra_data.split(',')
      for d in extra_data:
        n,path=d.split(':')
        ds.append(self._data(n, path, 'dev+'))

    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.tsv', 'test')
        ]
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev'):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    if type_name=='test':
      examples = ExampleSet([ExampleInstance((l[1], l[2])) for l in data[1:]])
    else:
      examples = ExampleSet([ExampleInstance((l[1],l[2]), self.label2id(l[3])) for l in data[1:]])

    predict_fn = self.get_predict_fn(examples)
    return EvalData(name, examples,
      metrics_fn = self.get_metrics_fn(), predict_fn = predict_fn)

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      return OrderedDict(accuracy=metric_accuracy(logits, labels))
    return metrics_fn

  def get_predict_fn(self, data):
    """Calcuate metrics based on prediction results"""
    def predict_fn(logits, output_dir, name, prefix):
      output = os.path.join(output_dir, 'pred-probs-{}-{}.tsv'.format(name, prefix))
      probs = softmax(logits, axis=-1)
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('sentence1\tsentence2\tnot_entailment\tentailment\n')
        for d,probs in zip(data, probs):
          fs.write(f'{d.segments[0]}\t{d.segments[1]}\t{probs[0]}\t{probs[1]}\n')
      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      preds = np.argmax(logits, axis=1)
      labels = self.get_labels()
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('index\tpredictions\n')
        for i,p in enumerate(preds):
          fs.write('{}\t{}\n'.format(i, labels[p]))
    return predict_fn

  def get_labels(self):
    """See base class."""
    return ["not_entailment", "entailment"]

@register_task('mrpc')
class MRPCTask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    input_src = os.path.join(self.data_dir, 'train.tsv')
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    examples = ExampleSet([ExampleInstance((l[3],l[4]), self.label2id(l[0])) for l in data[1:]]) # if l[3] in ['slate']])
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', "dev.tsv", 'dev'),
        ]
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.tsv', 'test')
        ]
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev'):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    predict_fn = self.get_predict_fn()
    if type_name=='test':
      examples = ExampleSet([ExampleInstance((l[3], l[4])) for l in data[1:]])
    else:
      examples = ExampleSet([ExampleInstance((l[3],l[4]), self.label2id(l[0])) for l in data[1:]])

    return EvalData(name, examples,
      metrics_fn = self.get_metrics_fn(), predict_fn = predict_fn)

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      return OrderedDict(accuracy= metric_accuracy(logits, labels),
          f1=metric_f1(logits, labels))
    return metrics_fn

  def get_predict_fn(self):
    """Calcuate metrics based on prediction results"""
    def predict_fn(logits, output_dir, name, prefix):
      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      preds = np.argmax(logits, axis=1)
      labels = self.get_labels()
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('index\tpredictions\n')
        for i,p in enumerate(preds):
          fs.write('{}\t{}\n'.format(i, labels[p]))
    return predict_fn

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

@register_task('qnli')
class QNLITask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    input_src = os.path.join(self.data_dir, 'train.tsv')
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    examples = ExampleSet([ExampleInstance((l[2],l[1]), self.label2id(l[3])) for l in data[1:]]) # if l[3] in ['slate']])
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', "dev.tsv", 'dev')
        ]
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.tsv', 'test')
        ]
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev'):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    predict_fn = self.get_predict_fn()
    if type_name=='test':
      examples = ExampleSet([ExampleInstance((l[2], l[1])) for l in data[1:]])
    else:
      examples = ExampleSet([ExampleInstance((l[2],l[1]), self.label2id(l[3])) for l in data[1:]])

    return EvalData(name, examples,
      metrics_fn = self.get_metrics_fn(), predict_fn = predict_fn)

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      return OrderedDict(accuracy=metric_accuracy(logits, labels))
    return metrics_fn

  def get_predict_fn(self):
    """Calcuate metrics based on prediction results"""
    def predict_fn(logits, output_dir, name, prefix):
      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      preds = np.argmax(logits, axis=1)
      labels = self.get_labels()
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('index\tpredictions\n')
        for i,p in enumerate(preds):
          fs.write('{}\t{}\n'.format(i, labels[p]))
    return predict_fn

  def get_labels(self):
    """See base class."""
    return ["not_entailment", "entailment"]

@register_task('cola')
class ColaTask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir
    if True:
      input_src = os.path.join(self.data_dir, 'train.tsv')
      assert os.path.exists(input_src), f"{input_src} doesn't exists"
      data = self._read_tsv(input_src)
      def get_hard_label(l):
        #return self.label2id(l[1])
        try:
          l = self.label2id(l[1])
        except Exception:
          import pdb
          pdb.set_trace()
        if l==0:
          return [1,0]
        else:
          return [0,1]

      train_examples = [ExampleInstance((l[3],), label=get_hard_label(l), domain_label=1) for l in data]

    self.train_split = train_examples[:-1000]
    self.train_dev = train_examples[-1000:]

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    examples = ExampleSet(self.train_dev + self.train_split) # if l[3] in ['slate']])
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen, label_type='float', training=True), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', "dev.tsv", 'dev'),
        ]

    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, label_type='int'), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.tsv', 'test'),
        ]

    if 'extra_data' in kwargs and kwargs['extra_data'] is not None:
      extra_data = kwargs['extra_data'].split(',')
      for d in extra_data:
        n,path=d.split(':')
        ds.append(self._data(n, path, 'test+'))

    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, label_type='int'), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev', ignore_metric=False):
    if isinstance(path, str):
      input_src = os.path.join(self.data_dir, path)
      assert os.path.exists(input_src), f"{input_src} doesn't exists"
      data = self._read_tsv(input_src)
      if type_name=='test':
        examples = ExampleSet([ExampleInstance((l[1], )) for l in data[1:]])
      else:
        examples = ExampleSet([ExampleInstance((l[3],), self.label2id(l[1])) for l in data])
    elif isinstance(path, ExampleSet):
      examples = path
    else:
      raise ValueError('Input type of path not supported')

    predict_fn = self.get_predict_fn(examples)
    return EvalData(name, examples,
      metrics_fn = self.get_metrics_fn(), predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['mcc'])

  def get_metrics_fn(self):
    def metric_fn(logits, labels):
      return OrderedDict(
      accuracy= metric_accuracy(logits, labels),
      mcc= metric_mcc(logits, labels))
    return metric_fn

  def get_predict_fn(self, data):
    """Calcuate metrics based on prediction results"""
    def predict_fn(logits, output_dir, name, prefix):
      output = os.path.join(output_dir, 'pred-probs-{}-{}.tsv'.format(name, prefix))
      probs = softmax(logits, axis=-1)
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('sentence\tlable_0\tlabel_1\n')
        for d,probs in zip(data, probs):
          fs.write(f'{d.segments[0]}\t{probs[0]}\t{probs[1]}\n')

      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      preds = np.argmax(logits, axis=-1)
      labels = self.get_labels()
      with open(output, 'w', encoding='utf-8') as fs:
        offset = 0
        sep = '\t'
        if name in ['test_id', 'test_od']:
          offset = 1
          sep = ','
          fs.write('Id,Label\n')
        else:
          fs.write('index\tpredictions\n')
        for i,p in enumerate(preds):
          fs.write('{}{}{}\n'.format(i+offset, sep, labels[p]))

    return predict_fn
  
  def get_labels(self):
    """See base class."""
    return ["0", "1"]

@register_task('sst-2')
class SST2Task(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    input_src = os.path.join(self.data_dir, 'train.tsv')
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    examples = ExampleSet([ExampleInstance((l[0],), self.label2id(l[1])) for l in data[1:]]) # if l[3] in ['slate']])
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', 'dev.tsv', 'dev')
        ]
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.tsv', 'test')
        ]
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev'):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    predict_fn = self.get_predict_fn()
    if type_name=='test':
      examples = ExampleSet([ExampleInstance((l[1], )) for l in data[1:]])
    elif type_name=='orig-test':
      examples = ExampleSet([ExampleInstance((l[1], ), self.label2id(l[3])) for l in data[1:]])
    else:
      examples = ExampleSet([ExampleInstance((l[0],), self.label2id(l[1])) for l in data[1:]])

    return EvalData(name, examples,
      metrics_fn = self.get_metrics_fn(), predict_fn = predict_fn)

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      return OrderedDict(accuracy= metric_accuracy(logits, labels))
    return metrics_fn
  
  def get_labels(self):
    """See base class."""
    return ["0", "1"]

@register_task('qqp')
class QQPTask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    input_src = os.path.join(self.data_dir, 'train.tsv')
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    examples = ExampleSet([ExampleInstance((l[3], l[4]), self.label2id(l[5])) for l in data[1:] if len(l)==6]) # if l[3] in ['slate']])
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', 'dev.tsv', 'dev')
        ]
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.tsv', 'test')
        ]
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      else:
        _size = dataset_size
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev'):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    predict_fn = self.get_predict_fn()
    if type_name=='test':
      examples = ExampleSet([ExampleInstance((l[-2], l[-1])) for l in data[1:]])
    else:
      examples = ExampleSet([ExampleInstance((l[3], l[4]), self.label2id(l[5])) for l in data[1:] if len(l)==6])

    return EvalData(name, examples,
      metrics_fn = self.get_metrics_fn(), predict_fn = predict_fn)

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      return OrderedDict(accuracy= metric_accuracy(logits, labels),
          f1=metric_f1(logits, labels))
    return metrics_fn
  
  def get_labels(self):
    """See base class."""
    return ["0", "1"]

@register_task('mnli')
class MNLITask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir

  def get_feature_fn(self, max_seq_len = 512, mask_gen = None):
    def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
      return example_to_feature(self.tokenizer, example, max_seq_len = max_seq_len, \
        rng = rng, mask_generator = mask_gen, ext_params = ext_params, **kwargs)
    return _example_to_feature

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    input_src = os.path.join(self.data_dir, 'train.tsv')
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    examples = [ExampleInstance((l[8], l[9]), self.label2id(l[-1])) for l in data[1:]] # if l[3] in ['slate']])
    examples = ExampleSet(examples)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('matched', 'dev_matched.tsv', 'dev'),
        self._data('mismatched', 'dev_mismatched.tsv', 'dev'),
        ]
   
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('matched', 'test_matched.tsv', 'test'),
        self._data('mismatched', 'test_mismatched.tsv', 'test'),
        ]
    
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def diagnostic_data(self, name, path, type_name='dev', ignore_metric=False):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    predict_fn = self.get_predict_fn()
    examples = ExampleSet([ExampleInstance((l[5], l[6]), self.label2id(l[7])) for l in data[1:]])

    def _metric_fn(logits, labels):
      return OrderedDict(
      accuracy= metric_accuracy(logits, labels),
      mcc= metric_mcc(logits, labels))
    return EvalData(name, examples,
      metrics_fn = _metric_fn, predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['mcc'])

  def anli_data(self, name, path, type_name='dev', ignore_metric=False):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    predict_fn = self.get_predict_fn()
    examples = ExampleSet([ExampleInstance((l[1], l[2]), self.label2id(l[3])) for l in data[1:]])

    def _metric_fn(logits, labels):
      return OrderedDict(
      accuracy= metric_accuracy(logits, labels))
    return EvalData(name, examples,
      metrics_fn = _metric_fn, predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy'])

  def _data(self, name, path, type_name = 'dev', ignore_metric=False):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self._read_tsv(input_src)
    predict_fn = self.get_predict_fn()
    if type_name=='test':
      examples = ExampleSet([ExampleInstance((l[8], l[9])) for l in data[1:]])
    else:
      examples = ExampleSet([ExampleInstance((l[8], l[9]), self.label2id(l[-1])) for l in data[1:]])

    return EvalData(name, examples,
      metrics_fn = self.get_metrics_fn(input_src), predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy'])

  def get_metrics_fn(self, input_src):
    """Calcuate metrics based on prediction results"""
    data = self._read_tsv(input_src)
    genres = [l[3] for l in data[1:]]
    def metrics_fn(logits, labels):
      metrics =  OrderedDict(accuracy= metric_accuracy(logits, labels))
      genres_predicts = defaultdict(list)
      for g,lg,lab in zip(genres,logits,labels):
        genres_predicts[g].append((lg, lab))
      for k in genres_predicts:
        logits_ = [x[0] for x in genres_predicts[k]]
        labels_ = [x[1] for x in genres_predicts[k]]
        acc = metric_accuracy(logits_, labels_)
        metrics[f'accuracy_{k}'] = acc
      return metrics
    return metrics_fn
  
  def get_labels(self):
    """See base class."""
    return ["contradiction", "neutral", "entailment"]

@register_task('anli')
class ANLITask(MNLITask):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    data_dir = data_dir.replace('/ANLI', '/MNLI')
    super().__init__(data_dir, tokenizer, args, **kwargs)

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    examples = []
    data_src = ['R1', 'R2', 'R3']
    for d in data_src:
      input_src = os.path.join(self.data_dir, f'anli_v0.1/{d}/train.tsv')
      data = self._read_tsv(input_src)
      examples += [ExampleInstance((l[1], l[2]), self.label2id(l[3])) for l in data[1:]]

    examples = ExampleSet(examples)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

