#
# Author: penhe@microsoft.com
# Date: 05/15/2019
#

from collections import defaultdict
import numpy as np
import pdb
import torch
import re
from ..optims import Fp16Optimizer,XAdam,ExpLossScaler
from ..utils import get_logger
logger=get_logger()

def xadam_factory(args, training_steps=None):
  def optimizer_fn(param_groups, max_grad_norm=None):
    with_radam = getattr(args, 'with_radam', False)
    opt_type = getattr(args, 'opt_type', None)
    optimizer = XAdam(param_groups,
            lr=args.learning_rate,
            b1=args.adam_beta1,
            b2=args.adam_beta2,
            lr_ends=args.lr_schedule_ends,
            e=args.epsilon,
            warmup=args.warmup_proportion if args.warmup_proportion<1 else args.warmup_proportion/training_steps,
            t_total=training_steps,
            schedule=args.lr_schedule,
            max_grad_norm = args.max_grad_norm if max_grad_norm is None else max_grad_norm,
            weight_decay_rate = args.weight_decay,
            with_radam = with_radam,
            opt_type = opt_type)
    return optimizer

  return optimizer_fn

def create_xoptimizer(model, args, num_train_steps=None, no_decay=['bias', 'LayerNorm.weight']):
  if args.fp16:
    loss_scaler = ExpLossScaler(scale_interval = args.scale_steps, init_scale=args.loss_scale)
  else:
    loss_scaler = None

  _no_decay = [x.strip() for x in getattr(args, 'no_decay', '').split('|') if len(x.strip())>0]
  if len(_no_decay)>0:
    no_decay = _no_decay

  opt_fn = xadam_factory(args, num_train_steps)
  
  named_params = list(model.named_parameters())
  type_groups = defaultdict(list)
  for n,p in named_params:
    key = ''
    if any(re.search(nd,n) for nd in no_decay):
      key += f'{str(p.dtype)}-nd'
    else:
      key += f'{str(p.dtype)}-d'
    type_groups[key].append((n,p))
  param_groups = []
  for key, params in type_groups.items():
    wd_theta = 0
    weight_decay = args.weight_decay
    if key.endswith('-nd'):
      weight_decay = 0

    group = dict(params=[],
      weight_decay_rate=weight_decay,
      wd_theta = wd_theta,
      names=[])
    for (n,p) in params:
      group['params'].append(p)
      group['names'].append(n)

    param_groups.append(group)
  lookahead_k = getattr(args, 'lookahead_k', -1)
  lookahead_alpha = getattr(args, 'lookahead_alpha', 0.5)
  optimizer = Fp16Optimizer(param_groups, opt_fn, loss_scaler, args.max_grad_norm, lookahead_k = lookahead_k,\
      lookahead_alpha = lookahead_alpha)

  return optimizer
