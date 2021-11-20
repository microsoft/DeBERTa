#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#

import os
import csv
import copy
from collections import OrderedDict,defaultdict,Sequence,Counter
import numpy as np
from ...utils import get_logger
from ...utils import xtqdm as tqdm
from ...data import example_to_feature
from .metrics import *

from ..models import SequenceClassificationModel
logger=get_logger()

__all__ = ['EvalData', 'Task']

class EvalData:
  def __init__(self, name, examples, metrics_fn=None, predict_fn=None, ignore_metric=False, critial_metrics=None):
    def accuracy_fn(logits, labels):
      return OrderedDict(accuracy= metric_accuracy(logits, labels))

    def default_pred_fn(logits, output_dir, name, prefix):
      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      preds = np.argmax(logits, axis=-1)
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('index\tpredictions\n')
        for i,p in enumerate(preds):
          fs.write('{}\t{}\n'.format(i, p))
    self.name = name
    self.data = examples
    self.ignore_metric = ignore_metric
    self.critial_metrics = critial_metrics
    self.metrics_fn = metrics_fn if metrics_fn is not None else accuracy_fn
    self.predict_fn = predict_fn if predict_fn is not None else default_pred_fn

  def __repr__(self):
    return f'{self.name}, {type(self.data)}: {len(self.data)}, {self.predict_fn}, {self.metrics_fn}'

class Task():
  _meta={}

  def __init__(self, tokenizer, args, **kwargs):
    self.tokenizer = tokenizer
    self.args = args
  
  def eval_data(self, **kwargs):
    raise NotImplementedError('Eval_data method not implemented yet.')

  def train_data(self, **kwargs):
    raise NotImplementedError('Eval_data method not implemented yet.')

  def test_data(self, **kwargs):
    raise NotImplementedError('Eval_data method not implemented yet.')

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  def label2id(self, labelstr):
    label_dict = {l:i for i,l in enumerate(self.get_labels())}
    return label_dict[labelstr] if labelstr in label_dict else -1

  def get_train_fn(self, *args, **kwargs):
    return None

  def get_eval_fn(self, *args, **kwargs):
    return None

  def get_pred_fn(self, *args, **kwargs):
    return None

  def get_loss_fn(self, *args, **kwargs):
    return None

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      return OrderedDict(accuracy= metric_accuracy(logits, labels))
    return metrics_fn

  def get_predict_fn(self):
    """Calcuate metrics based on prediction results"""
    def predict_fn(logits, output_dir, name, prefix):
      output=os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
      preds = np.argmax(logits, axis=-1)
      labels = self.get_labels()
      with open(output, 'w', encoding='utf-8') as fs:
        fs.write('index\tpredictions\n')
        for i,p in enumerate(preds):
          fs.write('{}\t{}\n'.format(i, labels[p]))

    return predict_fn

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def get_feature_fn(self, max_seq_len = 512, mask_gen = None, label_type='int', training=False):
    tokenizer = self.tokenizer
    def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
      return example_to_feature(tokenizer, example, max_seq_len = max_seq_len, \
        rng = rng, mask_generator = mask_gen, ext_params = ext_params, label_type=label_type, **kwargs)
    return _example_to_feature

  def get_model_class_fn(self):
    return SequenceClassificationModel.load_model
  
  @classmethod
  def add_arguments(cls, parser):
    """Add task specific arguments
      e.g. parser.add_argument('--data_dir', type=str, help='The path of data directory.')
    """
    parser.add_argument('--task_example_arg', type=str, default=None, help='An example task specific argument')

    return parser
