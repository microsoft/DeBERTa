#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Pengcheng He (penhe@microsoft.com)
# Date: 05/15/2019
#

""" Arguments for optimizer
"""
import argparse
from ..utils import boolean_string

__all__ = ['get_args']
def get_args():
  parser=argparse.ArgumentParser(add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  group = parser.add_argument_group(title='Optimizer', description='Parameters for the distributed optimizer')
  group.add_argument('--fp16',
            default=False,
            type=boolean_string,
            help="Whether to use 16-bit float precision instead of 32-bit")

  group.add_argument('--loss_scale',
            type=float, default=16384,
            help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

  group.add_argument('--scale_steps',
            type=int, default=250,
            help='The steps to wait to increase the loss scale.')

  group.add_argument('--lookahead_k',
            default=-1,
            type=int,
            help="lookahead k parameter")

  group.add_argument('--lookahead_alpha',
            default=0.5,
            type=float,
            help="lookahead alpha parameter")

  group.add_argument('--with_radam',
            default=False,
            type=boolean_string,
            help="whether to use RAdam")

  group.add_argument('--opt_type',
            type=str.lower,
            default='adam',
            choices=['adam', 'admax'],
            help="The optimizer to be used.")

  group.add_argument("--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
              "E.g., 0.1 = 10%% of training.")

  group.add_argument("--lr_schedule_ends",
            default=0,
            type=float,
            help="The ended learning rate scale for learning rate scheduling")

  group.add_argument("--lr_schedule",
            default='warmup_linear',
            type=str,
            help="The learning rate scheduler used for traning. " +
              "E.g. warmup_linear, warmup_linear_shift, warmup_cosine, warmup_constant. Default, warmup_linear")

  group.add_argument("--max_grad_norm",
            default=1,
            type=float,
            help="The clip threshold of global gradient norm")

  group.add_argument("--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.")

  group.add_argument("--epsilon",
            default=1e-6,
            type=float,
            help="epsilon setting for Adam.")

  group.add_argument("--adam_beta1",
            default=0.9,
            type=float,
            help="The beta1 parameter for Adam.")

  group.add_argument("--adam_beta2",
            default=0.999,
            type=float,
            help="The beta2 parameter for Adam.")

  group.add_argument('--weight_decay',
            type=float,
            default=0.01,
            help="The weight decay rate")

  return parser

