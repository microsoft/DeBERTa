#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Pengcheng He (penhe@microsoft.com)
# Date: 05/15/2019
#

""" Optimizer
"""

import math
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from torch import distributed as dist
import pdb
from .lr_schedulers import SCHEDULES
from ..utils import get_logger

def adamw(data,
    out_data,
    next_m,
    next_v,
    grad,
    lr,
    beta1,
    beta2,
    eps,
    grad_scale, #combined_scale, g = g/scale
    step,
    eps_mode = 1, #self.eps_mode, esp inside sqrt:0, outside: 1, only update with momentum: 2
    bias_correction = 0,
    weight_decay = 0):
  if bias_correction > 0:
    lr *= bias_correction
  beta1_ = 1 - beta1
  beta2_ = 1 - beta2
  grad = grad.float()
  if grad_scale != 1:
    grad *= 1/grad_scale
  next_m.mul_(beta1).add_(beta1_, grad)
  # admax
  admax = eps_mode>>4
  eps_mode = eps_mode&0xF
  if admax > 0:
    torch.max(next_v.mul_(beta2), grad.abs().to(next_v), out=next_v)
    update = next_m/(next_v+eps)
  else:
    next_v.mul_(beta2).addcmul_(beta2_, grad, grad)
    if eps_mode == 0:
      update = (next_m)*(next_v+eps).rsqrt()
    elif eps_mode == 1:
      update = (next_m)/(next_v.sqrt()+eps)
    else: #=2
      update = next_m.clone()
  if weight_decay>0:
    update.add_(weight_decay, data)

  data.add_(-lr, update)
  if (out_data is not None) and len(out_data)>0:
    out_data.copy_(data)

class XAdam(Optimizer):
  """Implements optimized version of Adam algorithm with weight decay fix.
  Params:
    lr: learning rate
    warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
    t_total: total number of training steps for the learning
      rate schedule, -1  means constant learning rate. Default: -1
    schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
    b1: Adams b1. Default: 0.9
    b2: Adams b2. Default: 0.999
    e: Adams epsilon. Default: 1e-6
    weight_decay_rate: Weight decay. Default: 0.01
    max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    with_radam: Whether to enable radam. Default: False
    radam_th: RAdam threshold for tractable variance. Default: 4
    opt_type: The type of optimizer, [adam, admax], default: adam
  """
  def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
         b1=0.9, b2=0.999, e=1e-8, weight_decay_rate=0.01,
         lr_ends = 0,
         max_grad_norm = 1.0,
         with_radam = False,
         radam_th = 4,
         opt_type=None,
         rank = -1):
    if not lr >= 0.0:
      raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
    if schedule not in SCHEDULES:
      raise ValueError("Invalid schedule parameter: {}".format(schedule))
    if not 0.0 <= warmup < 1.0 and not warmup == -1:
      raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
    if not 0.0 <= b1 < 1.0:
      raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
    if not 0.0 <= b2 < 1.0:
      raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
    if not e >= 0.0:
      raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
    self.defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
            b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
            lr_ends = lr_ends,
            max_grad_norm=max_grad_norm,
            with_radam = with_radam, radam_th = radam_th)
    self.opt_type = opt_type.lower() if opt_type is not None else ""
    self.rank = rank
    super().__init__(params, self.defaults)

  def step(self, grad_scale = 1, lr_scale = 1):
    """Performs a single optimization step.

    Arguments:
      grad_scale: divid grad by grad_scale
      lr_scale: scale learning rate by bs_scale
    """
    if 'global_step' not in self.state:
      self.state['global_step'] = 0
    for group in self.param_groups:
      lr_sch = self.get_group_lr_sch(group, self.state['global_step'])
      if group['rank'] == self.rank or group['rank']<0 or self.rank<0:
        for param in group['params']:
          self.update_param(group, param, grad_scale, lr_scale)

    self.state['global_step'] += 1
    self.last_grad_scale = grad_scale
    handels = []
    for group in self.param_groups:
      if group['rank']>=0 and self.rank>=0:
        # sync
        for param in group['params']:
          out_p = param.out_data if hasattr(param, 'out_data') and (param.out_data is not None) else None
          if out_p is not None:
            h = torch.distributed.broadcast(out_p, group['rank'], async_op=True)
          else:
            h = torch.distributed.broadcast(param.data, group['rank'], async_op=True)
          handels.append(h)

    for h in handels:
      if h is not None:
        h.wait()

    return lr_sch

  def get_group_lr_sch(self, group, steps):
    if group['t_total'] > 0:
      schedule_fct = SCHEDULES[group['schedule']]
      lr_scheduled = schedule_fct(steps, group['t_total'], group['warmup'], group['lr_ends'])
    else:
      lr_scheduled = 1
    return lr_scheduled

  def update_param(self, group, param, grad_scale, lr_scale):
    grad = param.grad
    if grad.is_sparse:
      raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
    state = self.get_state(param)
    lr_sch = self.get_group_lr_sch(group, state['step'])
    lr = group['lr'] * lr_scale *lr_sch
    next_m, next_v = state['next_m'], state['next_v']
    beta1, beta2 = group['b1'], group['b2']
    state['step'] += 1

    # Support for RAdam
    t = (state['step']-1) + 1
    eps_mode = 1
    if group['with_radam']:
      rou_ = 2/(1-beta2) - 1
      rou_t = rou_ - 2*t/(beta2**-t - 1)
      bias_c = 1/(1-beta1**t)
      if rou_t > group['radam_th']:
        bias_c *= math.sqrt(1 - beta2**t)
        bias_c *= math.sqrt(((rou_t - 4)*(rou_t - 2)*rou_)/((rou_ - 4)*(rou_ - 2)*rou_t))
      else:
        eps_mode = 2
        bias_c = 0
      lr *= bias_c

    if self.opt_type == 'admax':
      eps_mode |= 0x10

    with torch.cuda.device(param.device.index):
      out_p = param.out_data if hasattr(param, 'out_data') and (param.out_data is not None) else None
      if out_p is None or out_p.dtype != grad.dtype:
        out_p = torch.tensor([], dtype=torch.float).to(param.data)
      
      weight_decay = group['weight_decay_rate']
      adamw(param.data,
                    out_p,
                    next_m,
                    next_v,
                    grad,
                    lr,
                    beta1,
                    beta2,
                    group['e'],
                    grad_scale, #combined_scale, g = g/scale
                    state['step'],
                    eps_mode, #self.eps_mode, esp inside sqrt:0, outside: 1, only update with momentum: 2
                    0, #bias_correction,
                    weight_decay)

      out_p = param.out_data if hasattr(param, 'out_data') and (param.out_data is not None) else None
      if out_p is not None and out_p.dtype != grad.dtype:
        out_p.copy_(param.data)

  def get_state(self, param):
    state = self.state[param]
    # State initialization
    if len(state) == 0:
      state['step'] = 0
      state['next_m'] = torch.zeros_like(param.data)
      state['next_v'] = torch.zeros_like(param.data)
    return state
