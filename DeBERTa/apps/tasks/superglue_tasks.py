# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#

from glob import glob
from collections import OrderedDict,defaultdict,Sequence,Counter
import copy
import math
import string
from nltk.tokenize import sent_tokenize
import pickle
from scipy.special import softmax
from multiprocessing import Pool, Queue, Process
from bisect import bisect
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
from ..models import MultiChoiceModel,ReCoRDQAModel

logger=get_logger()

__all__ = ['BoolQTask', 'ReCoRDTask', 'CBTask', 'MultiRCTask', 'WiCTask', 'SRTETask', 'COPATask']

class SuperGLUE(Task):
  def __init__(self, tokenizer, **kwargs):
    super().__init__(tokenizer, **kwargs)

  def get_predict_fn(self):
    def predict_fn(logits, output_dir, name, prefix):
      output=os.path.join(output_dir, 'submit-{}-{}.jsonl'.format(name, prefix))
      preds = np.argmax(logits, axis=-1)
      labels = self.get_labels()
      with open(output, 'w', encoding='utf-8') as fs:
        for i,p in enumerate(preds):
          ans = {"idx": i, "label": labels[p]}
          fs.write(json.dumps(ans) + '\n')
    return predict_fn

# TODO: Long sequence handling
@register_task(name='BoolQ', desc="BoolQ - question answering dataset for yes/no questions.")
class BoolQTask(SuperGLUE):
  def __init__(self, data_dir, tokenizer, **kwargs):
    super().__init__(tokenizer, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    train = self.load_data(os.path.join(self.data_dir, 'train.jsonl'))
    examples = ExampleSet(train)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', 'val.jsonl', 'dev'),
        ]
   
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.jsonl', 'test')
        ]
    
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev', ignore_metric=False, max_examples=None, shuffle=False):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self.load_data(input_src, max_examples=max_examples, shuffle=shuffle)
    examples = ExampleSet(data)
    predict_fn = self.get_predict_fn()
    metrics_fn = self.get_metrics_fn()
    return EvalData(name, examples,
      metrics_fn = metrics_fn, predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy'])

  def get_feature_fn(self, max_seq_len = 512, mask_gen = None):
    def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
      return self.example_to_feature(self.tokenizer, example, max_seq_len = max_seq_len, \
        rng = rng, mask_generator = mask_gen, ext_params = ext_params, **kwargs)
    return _example_to_feature

  def get_labels(self):
    """See base class."""
    return ['false', 'true']

  def load_data(self, path, max_seq_len=512, max_examples=None, shuffle=False):
    with open(path) as fs:
      data = [json.loads(l) for l in fs]
    examples=[]
    for d in data:
      passage = self.tokenizer.tokenize(d['passage'].strip())
      question = self.tokenizer.tokenize(d['question'].strip())
      label = None if 'label' not in d else self.label2id(str(d['label']).lower())
      examples.append(ExampleInstance(segments=[passage, question], label=label))
    
    def get_stats(l):
      return f'Max={max(l)}, min={min(l)}, avg={np.mean(l)}'
    ctx_token_size = [len(e.segments[0]) for e in examples]
    q_token_size = [len(e.segments[1]) for e in examples]
    total_size = [len(e.segments[0]) + len(e.segments[1]) for e in examples]
    logger.info(f'Context statistics: {get_stats(ctx_token_size)}, long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}')
    logger.info(f'question statistics: {get_stats(q_token_size)}')
    logger.info(f'Total statistics: {get_stats(total_size)}, long={len([t for t in total_size if t>500])}')

    return examples

  def example_to_feature(self, tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, label_type='int', **kwargs):
    if not rng:
      rng = random
    max_num_tokens = max_seq_len - len(example.segments) - 1
    features = OrderedDict()
    tokens = ['[CLS]']
    type_ids = [0]
    segments = [example.segments[0], example.segments[1]]
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

    for f in features:
      features[f] = torch.tensor(features[f], dtype=torch.int)
    if example.label is not None: # and example.label[0]>=0 and example.label[1]>=0:
      label_type = torch.int if label_type=='int' else torch.float
      features['labels'] = torch.tensor(example.label, dtype=label_type)
    return features

@register_task('CB')
class CBTask(SuperGLUE):
  def __init__(self, data_dir, tokenizer, **kwargs):
    super().__init__(tokenizer, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    train = self.load_data(os.path.join(self.data_dir, 'train.jsonl'))
    examples = ExampleSet(train)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', 'val.jsonl', 'dev'),
        ]
   
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.jsonl', 'test')
        ]
    
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev', ignore_metric=False, max_examples=None, shuffle=False):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self.load_data(input_src, max_examples=max_examples, shuffle=shuffle)
    examples = ExampleSet(data)
    predict_fn = self.get_predict_fn()
    metrics_fn = self.get_metrics_fn(examples)
    return EvalData(name, examples,
      metrics_fn = metrics_fn, predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['f1'])

  def get_feature_fn(self, max_seq_len = 512, mask_gen = None):
    def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
      return self.example_to_feature(self.tokenizer, example, max_seq_len = max_seq_len, \
        rng = rng, mask_generator = mask_gen, ext_params = ext_params, **kwargs)
    return _example_to_feature

  def get_labels(self):
    """See base class."""
    return ["contradiction", "neutral", "entailment"]

  def load_data(self, path, max_seq_len=512, max_examples=None, shuffle=False):
    with open(path) as fs:
      data = [json.loads(l) for l in fs]
    examples=[]
    for d in data:
      passage = self.tokenizer.tokenize(d['premise'].strip())
      question = self.tokenizer.tokenize(d['hypothesis'].strip())
      label = None if 'label' not in d else self.label2id(d['label'])
      examples.append(ExampleInstance(segments=[passage, question], label=label))
    
    def get_stats(l):
      return f'Max={max(l)}, min={min(l)}, avg={np.mean(l)}'
    ctx_token_size = [len(e.segments[0]) for e in examples]
    q_token_size = [len(e.segments[1]) for e in examples]
    total_size = [len(e.segments[0]) + len(e.segments[1]) for e in examples]
    logger.info(f'Premise statistics: {get_stats(ctx_token_size)}, long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}')
    logger.info(f'Hypothesis statistics: {get_stats(q_token_size)}')
    logger.info(f'Total statistics: {get_stats(total_size)}, long={len([t for t in total_size if t>500])}')

    return examples

  def get_metrics_fn(self, examples):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      return OrderedDict(
          accuracy = metric_accuracy(logits, labels),
          f1 = metric_macro_f1(logits, labels, [0, 1, 2])
          )
    return metrics_fn

  def example_to_feature(self, tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, label_type='int', **kwargs):
    if not rng:
      rng = random
    max_num_tokens = max_seq_len - len(example.segments) - 1
    features = OrderedDict()
    tokens = ['[CLS]']
    type_ids = [0]
    segments = [example.segments[0], example.segments[1]]
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

    for f in features:
      features[f] = torch.tensor(features[f], dtype=torch.int)
    if example.label is not None: # and example.label[0]>=0 and example.label[1]>=0:
      label_type = torch.int if label_type=='int' else torch.float
      features['labels'] = torch.tensor(example.label, dtype=label_type)
    return features

# TODO: handle long sequence 
@register_task('MultiRC')
class MultiRCTask(SuperGLUE):
  def __init__(self, data_dir, tokenizer, **kwargs):
    super().__init__(tokenizer, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    train = self.load_data(os.path.join(self.data_dir, 'train.jsonl'))
    examples = ExampleSet(train)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', 'val.jsonl', 'dev'),
        ]
   
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.jsonl', 'test')
        ]
    
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev', ignore_metric=False, max_examples=None, shuffle=False):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self.load_data(input_src, max_examples=max_examples, shuffle=shuffle)
    examples = ExampleSet(data)
    predict_fn = self.get_predict_fn(examples)
    metrics_fn = self.get_metrics_fn(examples)
    return EvalData(name, examples,
      metrics_fn = metrics_fn, predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['f1'])

  def get_predict_fn(self, examples):
    def predict_fn(logits, output_dir, name, prefix):
      results = defaultdict(list)
      for el,e in zip(logits, examples):
        results[e.qidx].append({"logit":el,
          "example": e, 'qid': e.qidx, 'doc_id': e.doc_id, 'ans_id': e.aidx})
      labels = self.get_labels()
      for qidx in results:
        qr = results[qidx]
        m = 0
        for answer in qr:
          label_id = np.argmax(answer['logit'])
          answer['label'] = labels[label_id]
      docs = defaultdict(list)
      for qid in results:
        ans = []
        for a in results[qid]:
          ans.append({
            'idx': a['ans_id'],
            'label': a['label']})
        docs[results[qid][0]['doc_id']].append({'idx': qid,
          'answers': ans})

      output=os.path.join(output_dir, 'submit-{}-{}.jsonl'.format(name, prefix))
      with open(output, 'w', encoding='utf-8') as fs:
        for doc_id in docs.keys():
          ans = {'idx': doc_id,
              'passage':{
                'questions': docs[doc_id]
                }
              }
          fs.write(json.dumps(ans) + '\n')
    return predict_fn

  def get_metrics_fn(self, examples):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      results = defaultdict(list)
      for el,e in zip(logits, examples):
        results[e.qidx].append([el, e])
      em = 0
      for qidx in results:
        qr = results[qidx]
        m = 0
        for l, e in qr:
          if np.argmax(l)==e.label:
            m += 1
        if m == len(qr):
          em += 1
      em = em/len(results)
      return OrderedDict(
          em = em,
          f1 = metric_f1(logits, labels)
          )
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
    with open(path) as fs:
      data = [json.loads(l) for l in fs]
    examples=[]
    for pid, d in enumerate(data):
      passage = self.tokenizer.tokenize(d['passage']['text'].strip())
      p = d['passage']
      for q in p['questions']:
        question = self.tokenizer.tokenize(q['question'].strip())
        qidx = q['idx']
        for ans in q['answers']:
          answer = self.tokenizer.tokenize(ans['text'].strip())
          label = None if 'label' not in ans else self.label2id(ans['label'])
          aidx = ans['idx']
          examples.append(ExampleInstance(segments=[passage, question + answer], label=label, aidx=aidx, qidx=qidx, doc_id=pid))
    
    def get_stats(l):
      return f'Max={max(l)}, min={min(l)}, avg={np.mean(l)}'
    ctx_token_size = [len(e.segments[0]) for e in examples]
    q_token_size = [len(e.segments[1]) for e in examples]
    #a_token_size = [len(e.segments[2]) for e in examples]
    total_size = [sum(len(s) for s in e.segments) for e in examples]
    logger.info(f'Premise statistics: {get_stats(ctx_token_size)}, long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}')
    logger.info(f'Question statistics: {get_stats(q_token_size)}')
    #logger.info(f'Answer statistics: {get_stats(a_token_size)}')
    logger.info(f'Total statistics: {get_stats(total_size)}, long={len([t for t in total_size if t>500])}')

    return examples

  def example_to_feature(self, tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, label_type='int', **kwargs):
    if not rng:
      rng = random
    max_num_tokens = max_seq_len - len(example.segments) - 1
    features = OrderedDict()
    tokens = ['[CLS]']
    type_ids = [0]
    segments = [example.segments[0], example.segments[1]] #, example.segments[2]]
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

    for f in features:
      features[f] = torch.tensor(features[f], dtype=torch.int)
    if example.label is not None: # and example.label[0]>=0 and example.label[1]>=0:
      label_type = torch.int if label_type=='int' else torch.float
      features['labels'] = torch.tensor(example.label, dtype=label_type)
    return features

@register_task('ReCoRD')
class ReCoRDTask(Task):
  def __init__(self, data_dir, tokenizer, **kwargs):
    super().__init__(tokenizer, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, debug=False, **kwargs):
    if debug:
      max_examples = 1000
    else:
      max_examples = None

    train = self.load_data(os.path.join(self.data_dir, 'train.jsonl'), max_examples=max_examples)
    examples = ExampleSet(train)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', 'val.jsonl', 'dev', max_examples=None),
        ]
   
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.jsonl', 'test')
        ]
    
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev', ignore_metric=False, max_examples=None, shuffle=False):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self.load_data(input_src, max_examples=max_examples, shuffle=shuffle)
    examples = ExampleSet(data)
    predict_fn = self.get_predict_fn(examples)
    metrics_fn = self.get_metrics_fn(examples)
    return EvalData(name, examples,
      metrics_fn = metrics_fn, predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['f1'])

  def get_metrics_fn(self, examples):
    """Calcuate metrics based on prediction results"""
    from .record_eval import evaluate as ReCoRD_Eval
    def metrics_fn(logits, labels):
      predictions = self.produce_answers(logits, examples)
      all_predictions = {id:[a['text'] for a in predictions[id]]for id in predictions}
      top1predictions = {id:predictions[id][0]['text'] for id in predictions}
      answers = {e.qid:e.answers for e in examples}
      metrics = ReCoRD_Eval(answers, top1predictions)
      return metrics
    return metrics_fn

  def get_predict_fn(self, examples, topk=5):
    def predict_fn(logits, output_dir, name, prefix):
      output=os.path.join(output_dir, 'submit-{}-{}.jsonl'.format(name, prefix))
      predictions = self.produce_answers(logits, examples)
      submit = [{'idx':id, "label": predictions[id][0]['text']} for id in predictions]
      with open(output, 'w', encoding='utf-8') as fs:
        for p in submit:
          fs.write(json.dumps(p) + '\n')
      output=os.path.join(output_dir, 'predictions-{}-{}.jsonl'.format(name, prefix))
      with open(output, 'w', encoding='utf-8') as fs:
        for id in predictions:
          p = {'idx':id, 'answers': predictions[id]}
          fs.write(json.dumps(p) + '\n')

    return predict_fn

  def produce_answers(self, logits, examples):
    results = {}
    logits = logits - np.log(1+np.exp(logits))
    for logit, example in zip(logits, examples):
      if example.qid not in results:
        results[example.qid] = []
      results[example.qid].append([logit, example])
    answers = {}

    for qid in results:
      result = results[qid]
      example = result[0][1]
      entity_scores = [[] for _ in example.entities]
      for l,e in result:
        for i,x in enumerate(e.entity_spans):
          entity_scores[x[0]].append(l[i])
      for k in range(len(entity_scores)):
        if len(entity_scores[k])>1:
          entity_scores[k] = np.mean(entity_scores[k])
        else:
          entity_scores[k] = entity_scores[k][0]
      
      entity_scores = np.exp(np.array(entity_scores)) - 0.5
      answer = []
      for score, entity in zip(entity_scores, example.entities):
        start,end = entity[0]
        answer.append({'start': start, 'end': end, 'text': example.passage[start: end], 'score': float(score)})
      answers[qid] = list(sorted(answer, key=lambda x:x['score'], reverse=True))

    return answers

  def get_feature_fn(self, max_seq_len = 512, mask_gen = None):
    def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
      return self.example_to_feature(self.tokenizer, example, max_seq_len = max_seq_len, \
        rng = rng, mask_generator = mask_gen, ext_params = ext_params, **kwargs)
    return _example_to_feature

  def get_labels(self):
    """See base class."""
    return [0, 1]

  def load_data(self, path, is_train=False, max_seq_len=512, stride=256, max_examples=None, shuffle=False):
    with open(path) as fs:
      data = [json.loads(l) for l in fs]
    if shuffle:
      random.shuffle(data)
    if max_examples is not None and max_examples>0:
      data = data[:max_examples]
    examples=[]
    for d in data:
      passage = d['passage']['text']
      entities = d['passage']['entities']
      qas = d['qas']
      passage_words = self.tokenizer.split_to_words(passage)
      p_wl = np.cumsum([len(c) for c in passage_words])
      passage_tokens = [self.tokenizer.tokenize(w) for w in passage_words]
      p_tokens = [t for wt in passage_tokens for t in wt]
      t_l = np.cumsum([len(t) for t in passage_tokens])
      def cspan_to_tspan(s,e, pw):
        if s==e:
          e = s+1
        ws = bisect(p_wl, s)
        we = bisect(p_wl, e)
        text = passage[s:e]
        if we==ws:
          we += 1
        decoded = ''.join(pw[ws:we])
        if text not in decoded:
          we += 1
          _decoded = ''.join(pw[ws:we])
          decoded = _decoded

        decoded2 = ''.join(pw[ws:we-1]) if we-ws>1 else ''
        
        # Normalize text
        text = self.tokenizer.decode(self.tokenizer.tokenize(text))
        try:
          span = [t_l[ws-1] if ws>0 else 0, t_l[we-1]]
        except:
          pdb.set_trace()
        decoded = self.tokenizer.decode(p_tokens[span[0]:span[1]])
        decoded2 = self.tokenizer.decode(p_tokens[span[0]:span[1]-1]) if span[1]-span[0]>1 else ''
        while text in decoded2:
          span[1] -= 1
          _decoded2 = self.tokenizer.decode(p_tokens[span[0]:span[1]-1]) if span[1]-span[0]>1 else ''
          decoded2 = _decoded2

        if text not in decoded or text in decoded2:
          logger.warning(f'[{text}]: [{decoded2}]: [{decoded}]')
        return span
      entity_spans=[]
      for en in entities:
        s = en['start']
        e = en['end'] + 1
        if s>=e:
          continue
        entity_spans.append([[s,e], cspan_to_tspan(s,e, passage_words)])
      entity_spans = sorted(entity_spans, key=lambda t:t[0][0])
      e_starts = [k[0][0] for k in entity_spans]
      for qa in qas:
        question = qa['query']
        placeholder='@placeholder'
        pl_pos = question.index(placeholder)
        q_tokens = self.tokenizer.tokenize(question[:pl_pos]) + [ '[MASK]' ] + self.tokenizer.tokenize(question[pl_pos+len(placeholder):])
        place_idx = q_tokens.index('[MASK]')
        qid = qa['idx']
        if 'answers' in qa:
          answers = qa['answers']
          labels = []
          for aw in answers:
            start = aw['start']
            end = aw['end'] + 1
            eid = bisect(e_starts, start)-1
            if entity_spans[eid][0][0]!=start or entity_spans[eid][0][1]!=end:
              pdb.set_trace()
            labels.append(eid)
        else:
          answers = None
          labels = None

        # only apply slid window to long context and question
        if len(p_tokens)+len(q_tokens)>max_seq_len-3:
          max_tokens = max_seq_len - 3 - len(q_tokens)
          sub_offset = 0
          ends = max_tokens
          assert stride < max_tokens, f'{stride}|{ends}'
          while sub_offset < len(p_tokens):
            sub_tokens = p_tokens[sub_offset:ends]
            sub_entities = []
            eid_map = {}
            for eid, entity in enumerate(entity_spans):
              entity_start, entity_end = entity[1]
              if entity_start>=sub_offset and entity_end<=ends:
                eid_map[eid] = len(sub_entities)
                sub_entities.append([eid, [entity_start - sub_offset, entity_end-sub_offset]])
            sub_labels = None
            if labels is not None:
              sub_labels = []
              for l in labels:
                if l in eid_map:
                  sub_labels.append(eid_map[l])
            if (not is_train) or len(sub_labels)>0:
              example = ExampleInstance(segments=[sub_tokens, q_tokens], label=sub_labels, entity_spans=sub_entities, passage=passage, entities = entity_spans, qid=qid, offset=sub_offset, answers=answers, placeholder = place_idx)
              examples.append(example)
            sub_offset = ends - stride
            ends = sub_offset + max_tokens
        else:
          sub_entities = [[eid, e[1]] for eid, e in enumerate(entity_spans)]
          example = ExampleInstance(segments=[p_tokens, q_tokens], label=labels, entity_spans=sub_entities, passage=passage, entities = entity_spans, qid=qid, offset=0, answers=answers, placeholder = place_idx)
          examples.append(example)

    def get_stats(l):
      return f'Max={max(l)}, min={min(l)}, avg={np.mean(l)}'
    ctx_token_size = [len(e.segments[0]) for e in examples]
    q_token_size = [len(e.segments[1]) for e in examples]
    entities = [len(e.entity_spans) for e in examples]
    entity_tokens = [e[1][1]-e[1][0] for ex in examples for e in ex.entity_spans]
    total_size = [len(e.segments[0]) + len(e.segments[1]) for e in examples]
    logger.info(f'Context statistics: {get_stats(ctx_token_size)}, long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}')
    logger.info(f'Question statistics: {get_stats(q_token_size)}')
    logger.info(f'Entities statistics: {get_stats(entities)}')
    logger.info(f'Entity tokens statistics: {get_stats(entity_tokens)}')
    logger.info(f'Total statistics: {get_stats(total_size)}, long={len([t for t in total_size if t>500])}')
    return examples

  def example_to_feature(self, tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, label_type='int', **kwargs):
    if not rng:
      rng = random
    max_num_tokens = max_seq_len - len(example.segments) - 1
    features = OrderedDict()
    tokens = ['[CLS]']
    type_ids = [0]
    segments = [example.segments[0], example.segments[1]]
    segments = _truncate_segments(segments, max_num_tokens, rng)

    for i,s in enumerate(segments):
      tokens.extend(s)
      tokens.append('[SEP]')
      type_ids.extend([i]*(len(s)+1))
    placeholder = example.placeholder + len(segments[0]) + 2
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

    # Max Entities 80
    # Max Enities spans 87 # 90
    max_entities = 110
    #max_entity_span = 110
    max_entity_span = 180
    entities = example.entity_spans
    assert len(entities)<=max_entities, f'Entities number {len(entities)} exceeds the maxium allowed entities {max_entities}'
    entity_indice = []
    for e, span in entities:
      token_pos = list(range(span[0]+1, span[1]+1))
      entity_indice.append(token_pos + [0]*(max_entity_span-len(token_pos)))
    for _ in range(len(entities), max_entities):
      entity_indice.append([0]*(max_entity_span))
    
    features['entity_indice'] = entity_indice
    features['placeholder'] = placeholder

    for f in features:
      features[f] = torch.tensor(features[f], dtype=torch.int)
    if example.label is not None: # and example.label[0]>=0 and example.label[1]>=0:
      labels = [0]*max_entities
      for l in example.label:
        labels[l] = 1
      label_type = torch.int if label_type=='int' else torch.float
      features['labels'] = torch.tensor(labels, dtype=label_type)
    return features

  def get_model_class_fn(self):
    def partial_class(*wargs, **kwargs):
      return ReCoRDQAModel.load_model(*wargs, **kwargs, na_pred=True)
    return partial_class

@register_task("COPA")
class COPATask(SuperGLUE):
  def __init__(self, data_dir, tokenizer, **kwargs):
    super().__init__(tokenizer, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    train = self.load_data(os.path.join(self.data_dir, 'train.jsonl'))
    examples = ExampleSet(train)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', 'val.jsonl', 'dev'),
        ]
   
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.jsonl', 'test')
        ]
    
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev', ignore_metric=False, max_examples=None, shuffle=False):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
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
      with open(output, 'w', encoding='utf-8') as fs:
        for i,p in enumerate(preds):
          o = {"idx": i, "label": int(p)}
          fs.write(json.dumps(o) + '\n')
    return predict_fn

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      metrics =  OrderedDict(accuracy= metric_multi_accuracy(logits, labels, 2))
      return metrics
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
    with open(path) as fs:
      data = [json.loads(l) for l in fs]
    examples=[]
    for d in data:
      passage = d['premise'].strip()
      question = d['question']
      opt1 = d['choice1'].strip()
      opt2 = d['choice2'].strip()
      label = None if 'label' not in d else d['label']
      if question.strip().lower() == "cause":
        examples.append(ExampleInstance(segments=[opt1 , passage], label=label==0 if label is not None else None))
        examples.append(ExampleInstance(segments=[opt2 , passage], label=label==1 if label is not None else None))
      else:
        examples.append(ExampleInstance(segments=[passage , opt1], label=label==0 if label is not None else None))
        examples.append(ExampleInstance(segments=[passage , opt2], label=label==1 if label is not None else None))
      #if question.strip().lower() == "cause":
      #  examples.append(ExampleInstance(segments=[opt1 + " caused " + passage], label=label==0 if label is not None else None))
      #  examples.append(ExampleInstance(segments=[opt2 + " daused " +  passage], label=label==1 if label is not None else None))
      #else:
      #  examples.append(ExampleInstance(segments=[passage + " caused " + opt1], label=label==0 if label is not None else None))
      #  examples.append(ExampleInstance(segments=[passage + " caused " + opt2], label=label==1 if label is not None else None))

    
    def get_stats(l):
      return f'Max={max(l)}, min={min(l)}, avg={np.mean(l)}'
    ctx_token_size = [len(e.segments[0]) for e in examples]
    #q_token_size = [len(e.segments[1]) for e in examples]
    total_size = [sum(len(s) for s in e.segments) for e in examples]
    logger.info(f'Premise statistics: {get_stats(ctx_token_size)}, long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}')
    #logger.info(f'Question statistics: {get_stats(q_token_size)}')
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
      segments = [self.tokenizer.tokenize(s) for s in segments]
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
      for f in features:
        features[f] = torch.tensor(features[f], dtype=torch.int)
      return features
    features = make_pair_feature(example.segments)

    if example.label is not None: # and example.label[0]>=0 and example.label[1]>=0:
      label_type = torch.int if label_type=='int' else torch.float
      features['labels'] = torch.tensor(example.label, dtype=label_type)
    return features

@register_task("WiC")
class WiCTask(SuperGLUE):
  def __init__(self, data_dir, tokenizer, **kwargs):
    super().__init__(tokenizer, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, debug=False, **kwargs):
    if debug:
      max_examples = 1000
    else:
      max_examples = None

    train = self.load_data(os.path.join(self.data_dir, 'train.jsonl'), max_examples=max_examples)
    examples = ExampleSet(train)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', 'val.jsonl', 'dev', max_examples=None),
        self._data('wordnet', 'train_weak.jsonl', 'dev'),
        ]
   
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
        self._data('test', 'test.jsonl', 'test')
        ]
    
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev', ignore_metric=False, max_examples=None, shuffle=False):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self.load_data(input_src, max_examples=max_examples, shuffle=shuffle)
    examples = ExampleSet(data)
    predict_fn = self.get_predict_fn()
    metrics_fn = self.get_metrics_fn()
    return EvalData(name, examples,
      metrics_fn = metrics_fn, predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy'])

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def binary_accuracy(logits, labels):
      pred = (logits>0).reshape(-1)
      match = pred==labels
      return np.sum(match)/len(labels)

    def metrics_fn(logits, labels):
      return OrderedDict(
          #accuracy = binary_accuracy(logits, labels),
          accuracy = metric_accuracy(logits, labels),
          )
    return metrics_fn

  def get_feature_fn(self, max_seq_len = 512, mask_gen = None):
    def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
      return self.example_to_feature(self.tokenizer, example, max_seq_len = max_seq_len, \
        rng = rng, mask_generator = mask_gen, ext_params = ext_params, **kwargs)
    return _example_to_feature

  def get_labels(self):
    """See base class."""
    return ['false', 'true']

  def load_data(self, path, is_train=False, max_seq_len=512, stride=256, max_examples=None, shuffle=False):
    with open(path) as fs:
      data = [json.loads(l) for l in fs]
    if shuffle:
      random.shuffle(data)
    if max_examples is not None and max_examples>0:
      data = data[:max_examples]
    examples=[]
    for d in data:
      s1 = d['sentence1']
      s2 = d['sentence2']
      word = d['word']
      label = str(d['label']).lower() if 'label' in d else None
      start1,end1 = d['start1'],d['end1']
      start2,end2 = d['start2'],d['end2']
      def tokenize_sentence(sent, start, end):
        words = self.tokenizer.split_to_words(sent)
        word_ends = np.cumsum([len(c) for c in words])
        word_tokens = [self.tokenizer.tokenize(w) for w in words]
        tokens = [t for wt in word_tokens for t in wt]
        word_token_ends = np.cumsum([len(t) for t in word_tokens])
        if start == end:
          end = sstart + 1
        ws = bisect(word_ends, start)
        we = bisect(word_ends, end)
        text = sent[start:end]
        if we==ws:
          we += 1
        span = [word_token_ends[ws-1] if ws>0 else 0, word_token_ends[we-1]]
        decoded = self.tokenizer.decode(tokens[span[0]:span[1]])
        decoded2 = self.tokenizer.decode(tokens[span[0]:span[1]-1]) if span[1]-span[0]>1 else ''
        while text in decoded2:
          span[1] -= 1
          _decoded2 = self.tokenizer.decode(tokens[span[0]:span[1]-1]) if span[1]-span[0]>1 else ''
          decoded2 = _decoded2

        if text not in decoded or text in decoded2:
          logger.warning(f'[{text}]: [{decoded2}]: [{_decoded2}]')
          pdb.set_trace()
        return tokens, span
      s1_tokens, s1_span = tokenize_sentence(s1, start1, end1)
      s2_tokens, s2_span = tokenize_sentence(s2, start2, end2)
      example = ExampleInstance(segments=[s1_tokens, s2_tokens], label=label, spans=[s1_span, s2_span])
      examples.append(example)

    def get_stats(l):
      return f'Max={max(l)}, min={min(l)}, avg={np.mean(l)}'
    s1_token_size = [len(e.segments[0]) for e in examples]
    s2_token_size = [len(e.segments[1]) for e in examples]
    span1 = [e.spans[0][1]-e.spans[0][0] for e in examples]
    total_size = [len(e.segments[0]) + len(e.segments[1]) for e in examples]
    logger.info(f'Sentence1 statistics: {get_stats(s1_token_size)}, long={len([t for t in s1_token_size if t > 500])}/{len(s1_token_size)}')
    logger.info(f'Sentence2 statistics: {get_stats(s2_token_size)}')
    logger.info(f'Word span statistics: {get_stats(span1)}')
    logger.info(f'Total statistics: {get_stats(total_size)}, long={len([t for t in total_size if t>500])}')
    return examples

  def example_to_feature(self, tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, label_type='int', **kwargs):
    if not rng:
      rng = random
    max_num_tokens = max_seq_len - len(example.segments) - 1
    features = OrderedDict()
    tokens = ['[CLS]']
    type_ids = [0]
    segments = [example.segments[0], example.segments[1]]
    segments = _truncate_segments(segments, max_num_tokens, rng)

    # Max word span 6
    max_word_span = 16
    words = example.spans
    word_indice = []
    for i,s in enumerate(segments):
      span = words[i]
      token_pos = list(range(span[0]+len(tokens), span[1] + len(tokens)))
      word_indice.append(token_pos + [0]*(max_word_span-len(token_pos)))

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

    features['word_spans'] = word_indice

    for f in features:
      features[f] = torch.tensor(features[f], dtype=torch.int)
    if example.label is not None: # and example.label[0]>=0 and example.label[1]>=0:
      features['labels'] = torch.tensor(self.label2id(example.label), dtype=torch.int)
    return features

  def get_model_class_fn(self):
    def partial_class(*wargs, **kwargs):
      return WiCModel.load_model(*wargs, **kwargs)
    return partial_class

def test_record_load_data():
  tokenizer = RoBERTaTokenizer('/mount/biglm/RoBERTa/base/dict.txt')
  data='/mount/biglm/bert/superglue/data/ReCoRD/train.jsonl'
  task = ReCoRDTask(os.path.dirname(data), tokenizer)
  task.load_data(data, is_train=True)

def test_copa_load_data():
  tokenizer = RoBERTaTokenizer('/mount/biglm/RoBERTa/base/dict.txt')
  data='/mount/biglm/bert/superglue/data/COPA/train.jsonl'
  task = COPATask(os.path.dirname(data), tokenizer)
  task.load_data(data)

def test_boolq_load_data():
  tokenizer = RoBERTaTokenizer('/mount/biglm/RoBERTa/base/dict.txt')
  data='/mount/biglm/bert/superglue/data/BoolQ/test.jsonl'
  task = BoolQTask(os.path.dirname(data), tokenizer)
  task.load_data(data)

def test_cb_load_data():
  tokenizer = RoBERTaTokenizer('/mount/biglm/RoBERTa/base/dict.txt')
  data='/mount/biglm/bert/superglue/data/CB/train.jsonl'
  task = CBTask(os.path.dirname(data), tokenizer)
  task.load_data(data)

def test_multirc_load_data():
  tokenizer = RoBERTaTokenizer('/mount/biglm/RoBERTa/base/dict.txt')
  data='/mount/biglm/bert/superglue/data/MultiRC/train.jsonl'
  task = MultiRCTask(os.path.dirname(data), tokenizer)
  examples = task.load_data(data)
  feature = task.example_to_feature(tokenizer, examples[0], max_seq_len=512)

  pdb.set_trace()

def test_wic_load_data():
  tokenizer = RoBERTaTokenizer('/mount/biglm/RoBERTa/base/dict.txt')
  data='/mount/biglm/bert/superglue/data/WiC/test.jsonl'
  task = WiCTask(os.path.dirname(data), tokenizer)
  examples = task.load_data(data)
  feature = task.example_to_feature(tokenizer, examples[0], max_seq_len=512)
  pdb.set_trace()
