# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/25/2020
#

"""DeBERTa finetuning runner."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
from ..deberta import tokenizers,load_vocab
from collections import OrderedDict, Mapping, Sequence
import argparse
import random
import time

import numpy as np
import math
import torch
import json
import shutil
from torch.utils.data import DataLoader
from ..utils import *
from ..utils import xtqdm as tqdm
from ..sift import AdversarialLearner,hook_sift_layer
from .tasks import load_tasks,get_task
from ._utils import merge_distributed, join_chunks

import pdb

from ..training import DistributedTrainer, initialize_distributed, batch_to, set_random_seed,kill_children
from ..data import DistributedBatchSampler, SequentialSampler, BatchSampler, AsyncDataLoader
from ..training import get_args as get_training_args
from ..optims import get_args as get_optims_args

def create_model(args, num_labels, model_class_fn):
  # Prepare model
  rank = getattr(args, 'rank', 0)
  init_model = args.init_model if rank<1 else None
  model = model_class_fn(init_model, args.model_config, num_labels=num_labels, \
      drop_out=args.cls_drop_out, \
      pre_trained = args.pre_trained)
  if args.fp16:
    model = model.half()

  logger.info(f'Total parameters: {sum([p.numel() for p in model.parameters()])}')
  return model

def train_model(args, model, device, train_data, eval_data, run_eval_fn, train_fn=None, loss_fn=None):
  total_examples = len(train_data)
  num_train_steps = int(len(train_data)*args.num_train_epochs / args.train_batch_size)
  logger.info("  Training batch size = %d", args.train_batch_size)
  logger.info("  Num steps = %d", num_train_steps)

  def data_fn(trainer):
    return train_data, num_train_steps, None

  def eval_fn(trainer, model, device, tag):
    results = run_eval_fn(trainer.args, model, device, eval_data, tag, steps=trainer.trainer_state.steps)
    eval_metric = np.mean([v[0] for k,v in results.items() if 'train' not in k])
    return eval_metric

  def _loss_fn(trainer, model, data):
    output = model(**data)
    loss = output['loss']
    return loss.mean(), data['input_ids'].size(0)

  def get_adv_loss_fn():
    adv_modules = hook_sift_layer(model, hidden_size=model.config.hidden_size, learning_rate=args.vat_learning_rate, init_perturbation=args.vat_init_perturbation)
    adv = AdversarialLearner(model, adv_modules)
    def adv_loss_fn(trainer, model, data):
      output = model(**data)
      logits = output['logits']
      loss = output['loss']
      if isinstance(logits, Sequence):
        logits = logits[-1]
      v_teacher = []
  
      t_logits = None
      if args.vat_lambda>0:
        def pert_logits_fn(model, **data):
          o = model(**data)
          logits = o['logits']
          if isinstance(logits, Sequence):
            logits = logits[-1]
          return logits
  
        loss += adv.loss(logits, pert_logits_fn, loss_fn = args.vat_loss_fn, **data)*args.vat_lambda
  
      return loss.mean(), data['input_ids'].size(0)
    return adv_loss_fn
  
  def _train_fn(args, model, device, data_fn, eval_fn, loss_fn):
    
    if loss_fn is None:
      loss_fn = get_adv_loss_fn() if args.vat_lambda>0 else _loss_fn
  
    trainer = DistributedTrainer(args, args.output_dir, model, device, data_fn, loss_fn = loss_fn, eval_fn = eval_fn, dump_interval = args.dump_interval)
    trainer.train()

  if train_fn is None:
    train_fn = _train_fn

  train_fn(args, model, device, data_fn = data_fn, eval_fn = eval_fn, loss_fn = loss_fn)

def calc_metrics(predicts, labels, eval_loss, eval_item, eval_results, args, name, prefix, steps, tag):
  tb_metrics = OrderedDict()
  result=OrderedDict()
  metrics_fn = eval_item.metrics_fn
  predict_fn = eval_item.predict_fn
  if metrics_fn is None:
    eval_metric = metric_accuracy(predicts, labels)
  else:
    metrics = metrics_fn(predicts, labels)
    result.update(metrics)
    critial_metrics = set(metrics.keys()) if eval_item.critial_metrics is None or len(eval_item.critial_metrics)==0 else eval_item.critial_metrics
    eval_metric = np.mean([v for k,v in metrics.items() if  k in critial_metrics])
  result['eval_loss'] = eval_loss
  result['eval_metric'] = eval_metric
  result['eval_samples'] = len(labels)
  if args.rank<=0:
    output_eval_file = os.path.join(args.output_dir, "eval_results_{}_{}.txt".format(name, prefix))
    with open(output_eval_file, 'w', encoding='utf-8') as writer:
      logger.info("***** Eval results-{}-{} *****".format(name, prefix))
      for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
        tb_metrics[f'{name}/{key}'] = result[key]

    if predict_fn is not None:
      predict_fn(predicts, args.output_dir, name, prefix)
    else:
      output_predict_file = os.path.join(args.output_dir, "predict_results_{}_{}.txt".format(name, prefix))
      np.savetxt(output_predict_file, predicts, delimiter='\t')
      output_label_file = os.path.join(args.output_dir, "predict_labels_{}_{}.txt".format(name, prefix))
      np.savetxt(output_label_file, labels, delimiter='\t')

  if not eval_item.ignore_metric:
    eval_results[name]=(eval_metric, predicts, labels)
  _tag = tag + '/' if tag is not None else ''
  def _ignore(k):
    ig = ['/eval_samples', '/eval_loss']
    for i in ig:
      if k.endswith(i):
        return True
    return False

def run_eval(args, model, device, eval_data, prefix=None, tag=None, steps=None):
  # Run prediction for full data
  prefix = f'{tag}_{prefix}' if tag is not None else prefix
  eval_results=OrderedDict()
  eval_metric=0
  no_tqdm = (True if os.getenv('NO_TQDM', '0')!='0' else False) or args.rank>0
  for eval_item in eval_data:
    name = eval_item.name
    eval_sampler = SequentialSampler(len(eval_item.data))
    batch_sampler = BatchSampler(eval_sampler, args.eval_batch_size)
    batch_sampler = DistributedBatchSampler(batch_sampler, rank=args.rank, world_size=args.world_size)
    eval_dataloader = DataLoader(eval_item.data, batch_sampler=batch_sampler, num_workers=args.workers)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predicts=[]
    labels=[]
    for batch in tqdm(AsyncDataLoader(eval_dataloader), ncols=80, desc='Evaluating: {}'.format(prefix), disable=no_tqdm):
      batch = batch_to(batch, device)
      with torch.no_grad():
        output = model(**batch)
      logits = output['logits'].detach()
      tmp_eval_loss = output['loss'].detach()
      if 'labels' in output:
        label_ids = output['labels'].detach().to(device)
      else:
        label_ids = batch['labels'].to(device)
      predicts.append(logits)
      labels.append(label_ids)
      eval_loss += tmp_eval_loss.mean().item()
      input_ids = batch['input_ids']
      nb_eval_examples += input_ids.size(0)
      nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    predicts = merge_distributed(predicts, len(eval_item.data))
    labels = merge_distributed(labels, len(eval_item.data))
    if isinstance(predicts, Sequence):
      for k,pred in enumerate(predicts):
        calc_metrics(pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), eval_loss, eval_item, eval_results, args, name + f'@{k}', prefix, steps, tag)
    else:
      calc_metrics(predicts.detach().cpu().numpy(), labels.detach().cpu().numpy(), eval_loss, eval_item, eval_results, args, name, prefix, steps, tag)

  return eval_results

def run_predict(args, model, device, eval_data, prefix=None):
  # Run prediction for full data
  eval_results=OrderedDict()
  eval_metric=0
  for eval_item in eval_data:
    name = eval_item.name
    eval_sampler = SequentialSampler(len(eval_item.data))
    batch_sampler = BatchSampler(eval_sampler, args.eval_batch_size)
    batch_sampler = DistributedBatchSampler(batch_sampler, rank=args.rank, world_size=args.world_size)
    eval_dataloader = DataLoader(eval_item.data, batch_sampler=batch_sampler, num_workers=args.workers)
    model.eval()
    predicts = []
    for batch in tqdm(AsyncDataLoader(eval_dataloader), ncols=80, desc='Evaluating: {}'.format(prefix), disable=args.rank>0):
      batch = batch_to(batch, device)
      with torch.no_grad():
        output = model(**batch)
      logits = output['logits']
      predicts.append(logits)
    predicts = merge_distributed(predicts, len(eval_item.data))
    if args.rank<=0:
      predict_fn = eval_item.predict_fn
      if predict_fn:
        if isinstance(predicts, Sequence):
          for k,pred in enumerate(predicts):
            output_test_file = os.path.join(args.output_dir, f"test_logits_{name}@{k}_{prefix}.txt")
            logger.info(f"***** Dump prediction results-{name}@{k}-{prefix} *****")
            logger.info("Location: {}".format(output_test_file))
            pred = pred.detach().cpu().numpy()
            np.savetxt(output_test_file, pred, delimiter='\t')
            predict_fn(pred, args.output_dir, name + f'@{k}', prefix)
        else:
          output_test_file = os.path.join(args.output_dir, "test_logits_{}_{}.txt".format(name, prefix))
          logger.info("***** Dump prediction results-{}-{} *****".format(name, prefix))
          logger.info("Location: {}".format(output_test_file))
          np.savetxt(output_test_file, predicts.detach().cpu().numpy(), delimiter='\t')
          predict_fn(predicts.detach().cpu().numpy(), args.output_dir, name, prefix)

def main(args):
  if not args.do_train and not args.do_eval and not args.do_predict:
    raise ValueError("At least one of `do_train` or `do_eval` or `do_predict` must be True.")
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  vocab_path, vocab_type = load_vocab(vocab_path = args.vocab_path, vocab_type = args.vocab_type, pretrained_id = args.init_model)
  tokenizer = tokenizers[vocab_type](vocab_path)
  task = get_task(args.task_name)(tokenizer = tokenizer, args=args, max_seq_len = args.max_seq_length, data_dir = args.data_dir)
  label_list = task.get_labels()

  eval_data = task.eval_data(max_seq_len=args.max_seq_length)
  logger.info("  Evaluation batch size = %d", args.eval_batch_size)
  if args.do_predict:
    test_data = task.test_data(max_seq_len=args.max_seq_length)
    logger.info("  Prediction batch size = %d", args.predict_batch_size)

  if args.do_train:
    train_data = task.train_data(max_seq_len=args.max_seq_length, debug=args.debug)
  model_class_fn = task.get_model_class_fn()
  model = create_model(args, len(label_list), model_class_fn)
  if args.do_train:
    with open(os.path.join(args.output_dir, 'model_config.json'), 'w', encoding='utf-8') as fs:
      fs.write(model.config.to_json_string() + '\n')
    shutil.copy(vocab_path, args.output_dir)
  logger.info("Model config {}".format(model.config))
  device = initialize_distributed(args)
  if not isinstance(device, torch.device):
    return 0
  model.to(device)
  run_eval_fn = task.get_eval_fn()
  loss_fn = task.get_loss_fn(args)
  if run_eval_fn is None:
    run_eval_fn = run_eval
  
  if args.do_eval:
    run_eval(args, model, device, eval_data, prefix=args.tag)

  if args.do_train:
    train_fn = task.get_train_fn(args, model)
    train_model(args, model, device, train_data, eval_data, run_eval_fn, loss_fn=loss_fn, train_fn = train_fn)

  if args.do_predict:
    run_predict(args, model, device, test_data, prefix=args.tag)

class LoadTaskAction(argparse.Action):
  _registered = False
  def __call__(self, parser, args, values, option_string=None):
    setattr(args, self.dest, values)
    if not self._registered:
      load_tasks(args.task_dir)
      all_tasks = get_task()
      if values=="*":
        for task in all_tasks.values():
          parser.add_argument_group(title=f'Task {task._meta["name"]}', description=task._meta["desc"])
        return

      assert values.lower() in all_tasks, f'{values} is not registed. Valid tasks {list(all_tasks.keys())}'
      task = get_task(values)
      group = parser.add_argument_group(title=f'Task {task._meta["name"]}', description=task._meta["desc"])
      task.add_arguments(group)
      type(self)._registered = True

def build_argument_parser():
  parser = argparse.ArgumentParser(parents=[get_optims_args(), get_training_args()], formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  ## Required parameters
  parser.add_argument("--task_dir",
            default=None,
            type=str,
            required=False,
            help="The directory to load customized tasks.")
  parser.add_argument("--task_name",
            default=None,
            type=str,
            action=LoadTaskAction,
            required=True,
            help="The name of the task to train. To list all registered tasks, use \"*\" as the name, e.g. \n"
            "\npython -m DeBERTa.apps.run --task_name \"*\" --help")

  parser.add_argument("--data_dir",
            default=None,
            type=str,
            required=False,
            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

  parser.add_argument("--output_dir",
            default=None,
            type=str,
            required=True,
            help="The output directory where the model checkpoints will be written.")

  ## Other parameters
  parser.add_argument("--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
              "Sequences longer than this will be truncated, and sequences shorter \n"
              "than this will be padded.")

  parser.add_argument("--do_train",
            default=False,
            action='store_true',
            help="Whether to run training.")

  parser.add_argument("--do_eval",
            default=False,
            action='store_true',
            help="Whether to run eval on the dev set.")

  parser.add_argument("--do_predict",
            default=False,
            action='store_true',
            help="Whether to run prediction on the test set.")

  parser.add_argument("--eval_batch_size",
            default=32,
            type=int,
            help="Total batch size for eval.")

  parser.add_argument("--predict_batch_size",
            default=32,
            type=int,
            help="Total batch size for prediction.")

  parser.add_argument('--init_model',
            type=str,
            help="The model state file used to initialize the model weights.")

  parser.add_argument('--model_config',
            type=str,
            help="The config file of bert model.")

  parser.add_argument('--cls_drop_out',
            type=float,
            default=None,
            help="The config file model initialization and fine tuning.")

  parser.add_argument('--tag',
            type=str,
            default='final',
            help="The tag name of current prediction/runs.")

  parser.add_argument('--debug',
            default=False,
            type=boolean_string,
            help="Whether to cache cooked binary features")

  parser.add_argument('--pre_trained',
            default=None,
            type=str,
            help="The path of pre-trained RoBERTa model")

  parser.add_argument('--vocab_type',
            default='gpt2',
            type=str,
            help="Vocabulary type: [spm, gpt2]")

  parser.add_argument('--vocab_path',
            default=None,
            type=str,
            help="The path of the vocabulary")

  parser.add_argument('--vat_lambda',
            default=0,
            type=float,
            help="The weight of adversarial training loss.")

  parser.add_argument('--vat_learning_rate',
            default=1e-4,
            type=float,
            help="The learning rate used to update pertubation")

  parser.add_argument('--vat_init_perturbation',
            default=1e-2,
            type=float,
            help="The initialization for pertubation")

  parser.add_argument('--vat_loss_fn',
            default="symmetric-kl",
            type=str,
            help="The loss function used to calculate adversarial loss. It can be one of symmetric-kl, kl or mse.")


  return parser

if __name__ == "__main__":
  parser = build_argument_parser()
  parser.parse_known_args()

  args = parser.parse_args()
  os.makedirs(args.output_dir, exist_ok=True)
  logger = set_logger(args.task_name, os.path.join(args.output_dir, 'training_{}.log'.format(args.task_name)))
  logger.info(args)
  try:
    main(args)
  except Exception as ex:
    try:
      logger.exception(f'Uncatched exception happened during execution.')
      import atexit
      atexit._run_exitfuncs()
    except:
      pass
    kill_children()
    os._exit(-1)
