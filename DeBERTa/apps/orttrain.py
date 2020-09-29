# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse
import random

import numpy as np
import torch
from ..deberta import GPT2Tokenizer, DebertaPreTrainedTokenizer
from ..onnx import ORTGlueTest
from ..utils import *
from .task_registry import tasks
from onnxruntime.capi._pybind_state import get_mpi_context_local_rank, get_mpi_context_local_size, get_mpi_context_world_rank, get_mpi_context_world_size

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

def main(args):
  os.makedirs(args.output_dir, exist_ok=True)
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  
  # load model based on task
  tokenizer = GPT2Tokenizer()
  processor = tasks[args.task_name.lower()](tokenizer = tokenizer, max_seq_len = args.max_seq_length, data_dir = args.data_dir)
  label_list = processor.get_labels()
  model_class_fn = processor.get_model_class_fn()
  model = create_model(args, len(label_list), model_class_fn)
  logger.info("Model config {}".format(model.config))
  
  # train with ORT
  test = ORTGlueTest()
  test.setUp(args)
  test.local_rank = get_mpi_context_local_rank()
  test.world_size = get_mpi_context_world_size()
  print("mpirun launch, local_rank / world_size: ", test.local_rank, test.world_size)
  os.environ['RANK'] = str(test.local_rank)
  os.environ['WORLD_SIZE'] = str(test.world_size)
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29501'
  test.model = model
  test.tokenizer = DebertaPreTrainedTokenizer()
  test.run_glue(task_name=args.task_name, fp16=False, use_new_api=True)

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
  parser.add_argument("--cache_dir",
            default=None,
            type=str,
            required=True,
            help="The directory to store the pretrained models downloaded from s3.")
  ## Other parameters, 
  parser.add_argument("--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
              "Sequences longer than this will be truncated, and sequences shorter \n"
              "than this will be padded.")
  parser.add_argument("--train_batch_size",
            default=32,
            type=int,
            help="Total batch size for training.")
  parser.add_argument("--eval_batch_size",
            default=32,
            type=int,
            help="Total batch size for eval.")
  parser.add_argument("--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.")
  parser.add_argument("--num_train_epochs",
            default=3.0,
            type=float,
            help="Total number of training epochs to perform.")
  parser.add_argument('--seed',
            type=int,
            default=1234,
            help="random seed for initialization")
  parser.add_argument('--fp16',
            default=False,
            type=boolean_string,
            help="Whether to use 16-bit float precision instead of 32-bit")
  parser.add_argument('--init_model',
            type=str,
            help="The model state file used to initialize the model weights.")
  parser.add_argument('--pre_trained',
            default=None,
            type=str,
            help="The path of pre-trained RoBERTa model")

  ## TBD: review params below
  parser.add_argument("--max_grad_norm",
            default=1,
            type=float,
            help="The clip threshold of global gradient norm")
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
  parser.add_argument('--accumulative_update',
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument('--loss_scale',
            type=float, default=256,
            help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
  parser.add_argument('--scale_steps',
            type=int, default=1000,
            help='The steps to wait to increase the loss scale.')
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
  parser.add_argument('--opt_type',
            type=str.lower,
            default='adam',
            choices=['adam', 'admax'],
            help="The optimizer to be used.")
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
    os._exit(-1)
