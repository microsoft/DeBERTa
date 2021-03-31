
from collections import OrderedDict,defaultdict,Sequence,Counter
import math
import numpy as np
import os
import pdb
import random
import torch
import ujson as json
from ...utils import xtqdm as tqdm
from ...utils import get_logger

from ..models import NERModel
from ...data import ExampleInstance, ExampleSet, DynamicDataset
from ...data.example import *
from ...data.example import _truncate_segments
from .task import EvalData, Task
from .task_registry import register_task

from seqeval import metrics as seq_metrics

__all__ = ['NERTask']

logger = get_logger()

@register_task(name="NER", desc="Named-entity recognition task")
class NERTask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir

  def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
    train = self.load_data(os.path.join(self.data_dir, 'train.txt'), max_seq_len=max_seq_len)
    examples = ExampleSet(train)
    if dataset_size is None:
      dataset_size = len(examples)*epochs
    return DynamicDataset(examples, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
    ds = [
        self._data('dev', 'valid.txt', 'dev', max_seq_len=max_seq_len),
        self._data('test', 'test.txt', 'test', max_seq_len=max_seq_len)
        ]
   
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def test_data(self,max_seq_len=512, dataset_size = None, **kwargs):
    """See base class."""
    ds = [
      self._data('test', 'test.txt', 'test', max_seq_len=max_seq_len)
      ]
    
    for d in ds:
      if dataset_size is None:
        _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn = self.get_feature_fn(max_seq_len=max_seq_len), dataset_size = _size, **kwargs)
    return ds

  def _data(self, name, path, type_name = 'dev', ignore_metric=False, max_examples=None, shuffle=False, max_seq_len=512):
    input_src = os.path.join(self.data_dir, path)
    assert os.path.exists(input_src), f"{input_src} doesn't exists"
    data = self.load_data(input_src, max_seq_len=max_seq_len, max_examples=max_examples, shuffle=shuffle)
    examples = ExampleSet(data)
    predict_fn = self.get_predict_fn(examples)
    metrics_fn = self.get_metrics_fn()
    return EvalData(name, examples,
      metrics_fn = metrics_fn, predict_fn = predict_fn, ignore_metric=ignore_metric, critial_metrics=['f1'])

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      preds = np.argmax(logits, axis=-1)
      label_names = self.get_labels()
      y_true = []
      y_pred = []
      for pred,label in zip(preds, labels):
        y_true.append([label_names[l] for l in label if l>=0])
        y_pred.append([label_names[p] for p,l in zip(pred, label) if l>=0])
      return OrderedDict(
        accuracy = seq_metrics.accuracy_score(y_true, y_pred),
        f1 = seq_metrics.f1_score(y_true, y_pred),
        precision = seq_metrics.precision_score(y_true, y_pred),
        recall = seq_metrics.recall_score(y_true, y_pred)
        )
    return metrics_fn

  def get_predict_fn(self, examples):
    """Calcuate metrics based on prediction results"""
    def predict_fn(logits, output_dir, name, prefix):
      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      preds = np.argmax(logits, axis=-1)
      labels = self.get_labels()
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('index\tpredictions\n')
        for i,(e,p) in enumerate(zip(examples,preds)):
          words = ''.join(e.sentence).split(' ')
          tokens = e.segments[0]
          bw = 0
          for w,t in zip(words,tokens):
            fs.write(f'{w} {labels[p[bw]]}\n')
            bw += len(t)
          fs.write('\n')

    return predict_fn

  def get_feature_fn(self, max_seq_len = 512, mask_gen = None):
    def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
      return self.example_to_feature(self.tokenizer, example, max_seq_len = max_seq_len, \
        rng = rng, mask_generator = mask_gen, ext_params = ext_params, **kwargs)
    return _example_to_feature

  def get_model_class_fn(self):
    def partial_class(*wargs, **kwargs):
      return NERModel.load_model(*wargs, **kwargs)
    return partial_class

  def get_labels(self):
    """See base class."""
    return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

  def load_data(self, path, max_seq_len=512, max_examples=None, shuffle=False):
    docs = self.extract_docs(path)
    examples=[]
    for doc in docs:
      merged_words = []
      merged_tokens = []
      merged_labels = []
      size = 0
      for sent in doc:
        words = [t[0] if i==0 else (' ' + t[0]) for i,t in enumerate(sent)]
        labels = [t[1] for t in sent]
        tokens = [self.tokenizer.tokenize(w) for w in words]
        l = sum(len(t) for t in tokens)
        if size+l > max_seq_len-2:
          examples.append(ExampleInstance(segments=[merged_tokens], label=merged_labels, sentence=merged_words))
          size = 0
          merged_words = []
          merged_tokens = []
          merged_labels = []
        size += l
        merged_words.extend(words)
        merged_tokens.extend(tokens)
        merged_labels.extend(labels)
      if size>0:
        examples.append(ExampleInstance(segments=[merged_tokens], label=merged_labels, sentence=merged_words))

    def get_stats(l):
      return f'Max={max(l)}, min={min(l)}, avg={np.mean(l)}'
    ctx_token_size = [sum(len(w) for w in  e.segments[0]) for e in examples]
    logger.info(f'Statistics: {get_stats(ctx_token_size)}, long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}')

    return examples

  def example_to_feature(self, tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, label_type='int', **kwargs):
    if not rng:
      rng = random
    max_num_tokens = max_seq_len - 2
    features = OrderedDict()
    tokens = ['[CLS]']
    target_labels = [-1]
    type_ids = [0]

    for i,w in enumerate(example.segments[0]):
      tokens.extend(w)
      type_ids.extend([0]*len(w))
      if example.label is not None:
        target_labels.append(self.label2id(example.label[i]))
        target_labels.extend([-1]*(len(w)-1))
    tokens.append('[SEP]')
    if example.label is not None:
      target_labels.extend([-1]*(max_seq_len-len(target_labels)))
    type_ids.append(0)
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
      features['labels'] = torch.tensor(target_labels, dtype=torch.int)
    return features

  def extract_docs(self, path):
    docs = []
    with open(path, 'r', encoding='utf-8') as fs:
      doc = []
      sent = []
      for line in fs:
        if line.startswith('-DOCSTART- '):
          if len(sent) > 0:
            doc.append(sent)
          sent = []
          if len(doc) > 0:
            docs.append(doc)
          doc = []
        elif line.strip() == '':
          if len(sent) > 0:
            doc.append(sent)
          sent = []
        else:
          tabs = line.split(' ')
          sent.append([tabs[0], tabs[-1].strip()])
    if len(sent) > 0:
      doc.append(sent)
      sent = []
    if len(doc) > 0:
      docs.append(doc)
      doc = []
    logger.info(f'Loaded {len(docs)} docs, {sum([len(d) for d in docs])} sentences.')
    return docs

def test_ner_load_data():
  tokenizer = GPT2Tokenizer()
  data='/mount/biglm/bert/NER/data/train.txt'
  task = NERTask(os.path.dirname(data), tokenizer)
  #docs = task.extract_docs(data)
  examples = task.load_data(data)
  feature = task.example_to_feature(tokenizer, examples[0], max_seq_len=512)
  pdb.set_trace()
