#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Pengcheng He (penhe@microsoft.com)
# Date: 05/15/2019
#

from collections import defaultdict
import numpy as np
import pdb
from functools import cmp_to_key
import torch
import re
from ..optims import Fp16Optimizer,XAdam,ExpLossScaler,get_world_size
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
            opt_type = opt_type,
            rank = args.rank)
    return optimizer

  return optimizer_fn

def create_xoptimizer(model, args, num_train_steps=None, no_decay=['bias', 'LayerNorm.weight']):
  if args.fp16:
    loss_scaler = ExpLossScaler(scale_interval = args.scale_steps, init_scale=args.loss_scale)
  else:
    loss_scaler = None

  distributed_optimizer = getattr(args, 'distributed_optimizer', True)
  max_distributed_groups = getattr(args, 'max_distributed_groups', 1000000)
  world_size = get_world_size()
  if world_size<=1:
    distributed_optimizer = False

  _no_decay = [x.strip() for x in getattr(args, 'no_decay', '').split('|') if len(x.strip())>0]
  if len(_no_decay)>0:
    no_decay = _no_decay

  opt_fn = xadam_factory(args, num_train_steps)
  
  named_params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
  param_size = [p.numel() for n,p in named_params]
  type_groups = defaultdict(list)
  if distributed_optimizer:
    num_groups = min(world_size, max_distributed_groups)
    max_group_size = (sum(param_size)+num_groups-1)//num_groups
    #max_group_size = max(64*1024*1024, max_group_size)
    #max_group_size = max_group_size//2
    max_group_size = (max_group_size//32)*32
    group_sizes = [0 for _ in range(num_groups)]
    group_ranks = [g*(world_size//num_groups) for g in range(num_groups)]
  else:
    # TODO: Fix inconsistent results with different group size
    max_group_size = max(64*1024*1024, max(param_size))
    num_groups = (sum(param_size)+max_group_size-1)//max_group_size
    group_sizes = [0 for _ in range(num_groups)]

  def get_smallest_group(group_sizes):
    return np.argmin([g+i/10000 for i,g in enumerate(group_sizes)])

  def chunk_into_pieces(param, max_size):
    num_chunks = param.numel()//max_size
    if num_chunks<2:
      return [param], [None]

    flat = param.view(-1)
    chunks=[]
    offsets = []
    for i in range(num_chunks-1):
      chunks.append(flat.narrow(0, i*max_size, max_size))
      offsets.append([i*max_size, max_size])
    i += 1
    chunks.append(flat.narrow(0, i*max_size, flat.size(0)-i*max_size))
    offsets.append([i*max_size, flat.size(0)-i*max_size])
    assert sum([c.numel() for c in chunks])==param.numel(), f'{param.numel()}: {offsets}'
    return chunks, offsets

  def param_cmp(x,y):
    n1,p1 = x
    n2,p2 = y
    if p1.numel() == p2.numel():
      if n1<n2:
        return -1
      elif n1>n2:
        return 1
      else:
        return 0
    else:
      return p1.numel() - p2.numel()

  def add_group(param_groups, group, group_id):
    if distributed_optimizer:
      group['rank'] = group_ranks[group_id]
    param_groups.append(group.copy())
    group['params'] = []
    group['names'] = []
    group['offset'] = None
    return get_smallest_group(group_sizes),group

  hard_reset = getattr(args, 'hard_reset', False)
  group_id = 0
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
    _hard_reset = False
    if key.endswith('-nd'):
      weight_decay = 0
    else:
      _hard_reset = hard_reset

    group = dict(params=[],
      weight_decay_rate=weight_decay,
      wd_theta = wd_theta,
      hard_reset = hard_reset,
      names=[],
      offset=None)
    params = sorted(params, key=cmp_to_key(param_cmp))
    for (n,p) in params:
      if p.numel() >= max_group_size:
        if len(group['params'])>0:
          group_id,group = add_group(param_groups, group, group_id)
        chunks, offsets = chunk_into_pieces(p, max_group_size)
        for chk, off in zip(chunks, offsets):
          group['params'].append(p)
          group['names'].append(n)
          group['offset'] = off
          group_sizes[group_id] += chk.numel()
          group_id,group = add_group(param_groups, group, group_id)
      else:
        group['params'].append(p)
        group['names'].append(n)
        group['offset'] = None
        group_sizes[group_id] += p.numel()
        if group_sizes[group_id]>=max_group_size:
          group_id,group = add_group(param_groups, group, group_id)
    if len(group['params'])>0:
      group_id,group = add_group(param_groups, group, group_id)

  lookahead_k = getattr(args, 'lookahead_k', -1)
  lookahead_alpha = getattr(args, 'lookahead_alpha', 0.5)
  optimizer = Fp16Optimizer(param_groups, opt_fn, loss_scaler, args.max_grad_norm, lookahead_k = lookahead_k,\
      lookahead_alpha = lookahead_alpha, rank=args.rank, distributed=distributed_optimizer)

  return optimizer
