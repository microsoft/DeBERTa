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
from collections import OrderedDict, Mapping, Sequence
import argparse
import random
import time

import numpy as np
import math
import torch
import json
from torch.utils.data import DataLoader
from ..deberta import GPT2Tokenizer
from ..utils import *
from ..utils import xtqdm as tqdm
from .task_registry import tasks

from ..training import DistributedTrainer, initialize_distributed, batch_to, set_random_seed,kill_children
from ..data import DistributedBatchSampler, SequentialSampler, BatchSampler, AsyncDataLoader

def create_model(args, num_labels, model_class_fn):
  # Prepare model
  rank = getattr(args, 'rank', 0)
  init_model = args.init_model if rank<1 else None
  model = model_class_fn(init_model, args.model_config, num_labels=num_labels, \
      drop_out=args.cls_drop_out, \
      pre_trained = args.pre_trained)
  if args.fp16:
    model = model.half()

  return model

def train_model(args, model, device, train_data, eval_data):
  total_examples = len(train_data)
  num_train_steps = int(len(train_data)*args.num_train_epochs / args.train_batch_size)
  logger.info("  Training batch size = %d", args.train_batch_size)
  logger.info("  Num steps = %d", num_train_steps)

  def data_fn(trainer):
    return train_data, num_train_steps, None

  def eval_fn(trainer, model, device, tag):
    results = run_eval(trainer.args, model, device, eval_data, tag, steps=trainer.trainer_state.steps)
    eval_metric = np.mean([v[0] for k,v in results.items() if 'train' not in k])
    return eval_metric

  def loss_fn(trainer, model, data):
    _, loss = model(**data)
    return loss.mean(), data['input_ids'].size(0)

  trainer = DistributedTrainer(args, model, device, data_fn, loss_fn = loss_fn, eval_fn = eval_fn, dump_interval = args.dump_interval)
  trainer.train()

def merge_distributed(data_list, max_len=None):
  merged = []
  def gather(data):
    data_chunks = [torch.zeros_like(data) for _ in range(args.world_size)]
    torch.distributed.all_gather(data_chunks, data)
    torch.cuda.synchronize()
    return data_chunks

  for data in data_list:
    if torch.distributed.is_initialized() and torch.distributed.get_world_size()>1:
      if isinstance(data, Sequence):
        data_chunks = []
        for d in data:
          chunks_ = gather(d)
          data_ = torch.cat(chunks_)
          data_chunks.append(data_)
        merged.append(data_chunks)
      else:
        data_chunks = gather(data)
        merged.extend(data_chunks)
    else:
      merged.append(data)
  if not isinstance(merged[0], Sequence):
    merged = torch.cat(merged)
    if max_len is not None:
      return merged[:max_len]
    else:
      return merged
  else:
    data_list=[]
    for d in zip(*merged):
      data = torch.cat(d)
      if max_len is not None:
        data = data[:max_len]
      data_list.append(data)
    return data_list

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
        logits, tmp_eval_loss = model(**batch)
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
    predicts=None
    for batch in tqdm(AsyncDataLoader(eval_dataloader), ncols=80, desc='Evaluating: {}'.format(prefix), disable=args.rank>0):
      batch = batch_to(batch, device)
      with torch.no_grad():
        logits, _ = model(**batch)
      if args.world_size>1:
        logits_all = [torch.zeros_like(logits) for _ in range(args.world_size)]
        torch.distributed.all_gather(logits_all, logits)
        torch.cuda.synchronize()
        logits = torch.cat(logits_all)
      logits = logits.detach().cpu().numpy()
      if predicts is None:
        predicts = np.copy(logits)
      else:
        predicts = np.append(predicts, logits, axis=0)
  
    predicts = predicts[:len(eval_item.data)]
    if args.rank<=0:
      output_test_file = os.path.join(args.output_dir, "test_logits_{}_{}.txt".format(name, prefix))
      logger.info("***** Dump prediction results-{}-{} *****".format(name, prefix))
      logger.info("Location: {}".format(output_test_file))
      np.savetxt(output_test_file, predicts, delimiter='\t')
      predict_fn = eval_item.predict_fn
      if predict_fn:
        predict_fn(predicts, args.output_dir, name, prefix)

def main(args):
  if not args.do_train and not args.do_eval and not args.do_predict:
    raise ValueError("At least one of `do_train` or `do_eval` or `do_predict` must be True.")
  os.makedirs(args.output_dir, exist_ok=True)
  task_name = args.task_name.lower()
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  tokenizer = GPT2Tokenizer()
  processor = tasks[task_name](tokenizer = tokenizer, max_seq_len = args.max_seq_length, data_dir = args.data_dir)
  label_list = processor.get_labels()

  eval_data = processor.eval_data(max_seq_len=args.max_seq_length)
  logger.info("  Evaluation batch size = %d", args.eval_batch_size)
  if args.do_predict:
    test_data = processor.test_data(max_seq_len=args.max_seq_length)
    logger.info("  Prediction batch size = %d", args.predict_batch_size)

  if args.do_train:
    train_data = processor.train_data(max_seq_len=args.max_seq_length, mask_gen = None, debug=args.debug)
  model_class_fn = processor.get_model_class_fn()
  model = create_model(args, len(label_list), model_class_fn)
  if args.do_train:
    with open(os.path.join(args.output_dir, 'model_config.json'), 'w', encoding='utf-8') as fs:
      fs.write(model.config.to_json_string() + '\n')
  logger.info("Model config {}".format(model.config))
  device = initialize_distributed(args)
  if not isinstance(device, torch.device):
    return 0
  model.to(device)
  if args.do_eval:
    run_eval(args, model, device, eval_data, prefix=args.tag)

  if args.do_train:
    train_model(args, model, device, train_data, eval_data)

  if args.do_predict:
    run_predict(args, model, device, test_data, prefix=args.tag)

def build_argument_parser():
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--task_name",
            default=None,
            type=str,
            required=True,
            help="The name of the task to train.")
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
  parser.add_argument("--train_batch_size",
            default=32,
            type=int,
            help="Total batch size for training.")
  parser.add_argument("--eval_batch_size",
            default=32,
            type=int,
            help="Total batch size for eval.")
  parser.add_argument("--predict_batch_size",
            default=32,
            type=int,
            help="Total batch size for prediction.")
  parser.add_argument("--max_grad_norm",
            default=1,
            type=float,
            help="The clip threshold of global gradient norm")
  parser.add_argument("--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.")
  parser.add_argument("--epsilon",
            default=1e-6,
            type=float,
            help="epsilon setting for Adam.")
  parser.add_argument("--adam_beta1",
            default=0.9,
            type=float,
            help="The beta1 parameter for Adam.")
  parser.add_argument("--adam_beta2",
            default=0.999,
            type=float,
            help="The beta2 parameter for Adam.")
  parser.add_argument("--num_train_epochs",
            default=3.0,
            type=float,
            help="Total number of training epochs to perform.")
  parser.add_argument("--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
              "E.g., 0.1 = 10%% of training.")
  parser.add_argument("--lr_schedule_ends",
            default=0,
            type=float,
            help="The ended learning rate scale for learning rate scheduling")
  parser.add_argument("--lr_schedule",
            default='warmup_linear',
            type=str,
            help="The learning rate scheduler used for traning. "
              "E.g. warmup_linear, warmup_linear_shift, warmup_cosine, warmup_constant. Default, warmup_linear")

  parser.add_argument("--local_rank",
            type=int,
            default=-1,
            help="local_rank for distributed training on gpus")

  parser.add_argument('--seed',
            type=int,
            default=1234,
            help="random seed for initialization")

  parser.add_argument('--accumulative_update',
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.")

  parser.add_argument('--fp16',
            default=False,
            type=boolean_string,
            help="Whether to use 16-bit float precision instead of 32-bit")

  parser.add_argument('--loss_scale',
            type=float, default=256,
            help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

  parser.add_argument('--scale_steps',
            type=int, default=1000,
            help='The steps to wait to increase the loss scale.')

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
  parser.add_argument('--weight_decay',
            type=float,
            default=0.01,
            help="The weight decay rate")

  parser.add_argument('--tag',
            type=str,
            default='final',
            help="The tag name of current prediction/runs.")

  parser.add_argument("--dump_interval",
            default=10000,
            type=int,
            help="Interval steps for generating checkpoint.")

  parser.add_argument('--lookahead_k',
            default=-1,
            type=int,
            help="lookahead k parameter")

  parser.add_argument('--lookahead_alpha',
            default=0.5,
            type=float,
            help="lookahead alpha parameter")

  parser.add_argument('--with_radam',
            default=False,
            type=boolean_string,
            help="whether to use RAdam")

  parser.add_argument('--opt_type',
            type=str.lower,
            default='adam',
            choices=['adam', 'admax'],
            help="The optimizer to be used.")

  parser.add_argument('--workers',
            type=int,
            default=2,
            help="The workers to load data.")

  parser.add_argument('--debug',
            default=False,
            type=boolean_string,
            help="Whether to cache cooked binary features")

  parser.add_argument('--pre_trained',
            default=None,
            type=str,
            help="The path of pre-trained RoBERTa model")
  return parser

if __name__ == "__main__":
  parser = build_argument_parser()
  args = parser.parse_args()
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
