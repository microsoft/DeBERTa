#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Pengcheng He (penhe@microsoft.com)
# Date: 05/15/2019
#

""" FP16 optimizer wrapper
"""

from collections import defaultdict
import numpy as np
import math
import torch
import pdb
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import ctypes

from ..utils import get_logger,boolean_string
logger=get_logger()

__all__ = ['Fp16Optimizer', 'ExpLossScaler', 'get_world_size']

def get_world_size():
  try:
    wd = dist.get_world_size()
    return wd
  except:
    return 1

def fused_norm(input):
  return torch.norm(input, p=2, dtype=torch.float32)

class OptParameter(torch.Tensor):
  def __new__(cls, data, out_data=None, grad=None, name=None):
    param = torch.Tensor._make_subclass(cls, data)
    param._xgrad = grad
    param.out_data = out_data
    param._name = name
    return param

  @property
  def name(self):
    return self._name

  @property
  def grad(self):
    return self._xgrad

  @grad.setter
  def grad(self, grad):
    self._xgrad = grad

class Fp16Optimizer(object):
  def __init__(self, param_groups, optimizer_fn, loss_scaler=None, grad_clip_norm = 1.0, lookahead_k = -1, lookahead_alpha = 0.5, rank=-1, distributed=False):
    # all parameters should on the same device
    groups = []
    original_groups = []
    self.rank = rank
    self.distributed = distributed
    if self.rank<0:
      self.distributed = False
    for group in param_groups:
      if 'offset' not in group:
        group['offset'] = None
      if ('rank' not in group) or (not self.distributed):
        group['rank'] = -1
        assert group['offset'] is None, f"{group['names']}: {group['offset']}"
      group_rank = group['rank']
      params = group['params'] # parameter
      if len(params) > 1:
        flattened_params = _flatten_dense_tensors([p.data for p in params])
        unflattend_params = _unflatten_dense_tensors(flattened_params, [p.data for p in params])
        for uf,p in zip(unflattend_params, params):
          p.data = uf
      else:
        flattened_params = params[0].data.view(-1)
        if group['offset'] is not None:
          start, length = group['offset']
          flattened_params = flattened_params.narrow(0, start, length)
      
      if params[0].dtype==torch.half:
        if self.rank == group_rank or (not self.distributed):
          master_params = flattened_params.clone().to(torch.float).detach_().to(flattened_params.device)
        else:
          master_params = flattened_params.clone().to(torch.float).detach_().cpu()
        group['params'] = [OptParameter(master_params, flattened_params, name='master')]
      else:
        group['params'] = [OptParameter(flattened_params, None, name='master')]
      
      o_group = defaultdict(list)
      o_group['names'] = group['names']
      o_group['params'] = params
      o_group['rank'] = group_rank
      o_group['offset'] = group['offset']

      group['names'] = ['master']

      original_groups.append(o_group)
      groups.append(group)
    self.param_groups = groups
    self.loss_scaler = loss_scaler
    self.optimizer = optimizer_fn(self.param_groups)
    self.original_param_groups = original_groups
    self.max_grad_norm = grad_clip_norm
    self.lookahead_k = lookahead_k
    self.lookahead_alpha = lookahead_alpha

  def backward(self, loss):
    if self.loss_scaler:
      loss_scale, loss, step_loss = self.loss_scaler.scale(loss)
    else:
      loss_scale = 1
      step_loss = loss.item()

    loss.backward()
    return loss_scale, step_loss

  def step(self, lr_scale, loss_scale = 1):
    grad_scale = self._grad_scale(loss_scale)
    if grad_scale is None or math.isinf(grad_scale):
      self.loss_scaler.update(False)
      return False

    if self.lookahead_k > 0:
      for p in self.param_groups:
        if 'la_count' not in p:
          # init
          #make old copy
          p['la_count'] = 0
          p['slow_params'] = [x.data.detach().clone().requires_grad_(False) for x in p['params']]
    self.optimizer.step(grad_scale, lr_scale)
    if self.lookahead_k > 0:
      for p in self.param_groups:
        p['la_count'] += 1
        if p['la_count'] == self.lookahead_k:
          p['la_count'] = 0
          for s,f in zip(p['slow_params'], p['params']):
            s.mul_(1-self.lookahead_alpha)
            s.add_(f.data.detach()*self.lookahead_alpha)
            f.data.copy_(s, non_blocking=True)
            if hasattr(f, 'out_data') and f.out_data is not None:
              f.out_data.copy_(f.data, non_blocking=True)

    if self.loss_scaler:
      self.loss_scaler.update(True)
    return True

  def zero_grad(self):
    for group, o_group in zip(self.param_groups, self.original_param_groups):
      for p in group['params']:
        p.grad = None
      for p in o_group['params']:
        p.grad = None

  def _grad_scale(self, loss_scale = 1):
    named_params = {}
    named_grads = {}
    for g in self.original_param_groups:
      for n,p in zip(g['names'], g['params']):
        named_params[n] = p
        named_grads[n] = p.grad if p.grad is not None else torch.zeros_like(p.data)
    
    wd = get_world_size()
    def _reduce(group):
      grads = [named_grads[n] for n in group]
      if len(grads)>1:
        flattened_grads = _flatten_dense_tensors(grads)
      else:
        flattened_grads = grads[0],view(-1)

      if wd > 1:
        flattened_grads /= wd
        handle = dist.all_reduce(flattened_grads, async_op=True)
      else:
        handle = None
      return flattened_grads, handle

    def _process_grad(group, flattened_grads, max_grad, norm):
      grads = [named_grads[n] for n in group]
      norm = norm.to(flattened_grads.device)
      norm = norm + fused_norm(flattened_grads)**2

      if len(grads) > 1:
        unflattend_grads = _unflatten_dense_tensors(flattened_grads, grads)
      else:
        unflattend_grads = [flattened_grads]

      for n,ug in zip(group, unflattend_grads):
        named_grads[n] = ug #.to(named_params[n].data)

      return max_grad, norm

    group_size = 0
    group = []
    max_size = 32*1024*1024
    norm = torch.zeros(1, dtype=torch.float)
    max_grad = 0
    
    all_grads = []
    for name in sorted(named_params.keys(), key=lambda x:x.replace('deberta.', 'bert.')):
      group.append(name)
      group_size += named_params[name].data.numel()
      if group_size>=max_size:
        flatten, handle = _reduce(group)
        all_grads.append([handle, flatten, group])
        group = []
        group_size = 0
    if group_size>0:
      flatten, handle = _reduce(group)
      all_grads.append([handle, flatten, group])
      group = []
      group_size = 0
    for h,fg,group in all_grads:
      if h is not None:
        h.wait()
      max_grad, norm = _process_grad(group, fg, max_grad, norm)

    norm = norm**0.5
    if torch.isnan(norm) or torch.isinf(norm) :#in ['-inf', 'inf', 'nan']:
      return None

    scaled_norm = norm.detach().item()/loss_scale
    grad_scale = loss_scale

    if self.max_grad_norm>0:
      scale = norm/(loss_scale*self.max_grad_norm)
      if scale>1:
        grad_scale *= scale

    for group, o_g in zip(self.param_groups, self.original_param_groups):
      grads = [named_grads[n] for n in o_g['names']]

      if len(grads) > 1:
        flattened_grads = _flatten_dense_tensors(grads)
      else:
        flattened_grads = grads[0].view(-1)
        if group['offset'] is not None:
          start, length = group['offset']
          flattened_grads = flattened_grads.narrow(0, start, length)
      if group['rank'] == self.rank or (not self.distributed):
        group['params'][0].grad = flattened_grads

    return grad_scale

class ExpLossScaler:
  def __init__(self, init_scale=2**16, scale_interval=1000):
    self.cur_scale = init_scale
    self.scale_interval = scale_interval
    self.invalid_cnt = 0
    self.last_scale = 0
    self.steps = 0
    self.down_scale_smooth = 0
    
  def scale(self, loss):
    assert self.cur_scale > 0, self.init_scale
    step_loss = loss.float().detach().item()
    if step_loss != 0 and math.isfinite(step_loss):
      loss_scale = self.cur_scale
    else:
      loss_scale = 1
    loss = loss.float()*loss_scale
    return (loss_scale, loss, step_loss)

  def update(self, is_valid = True):
    if not is_valid:
      self.invalid_cnt += 1
      if self.invalid_cnt>self.down_scale_smooth:
        self.cur_scale /= 2
        self.cur_scale = max(self.cur_scale, 1)
        self.last_scale = self.steps
    else:
      self.invalid_cnt = 0
      if self.steps - self.last_scale>self.scale_interval:
        self.cur_scale *= 2
        self.last_scale = self.steps
    self.steps += 1

  def state_dict(self):
    state = defaultdict(float)
    state['steps'] = self.steps
    state['invalid_cnt'] = self.invalid_cnt
    state['cur_scale'] = self.cur_scale
    state['last_scale'] = self.last_scale
    return state

  def load_state_dict(self, state):
    self.steps = state['steps']
    self.invalid_cnt = state['invalid_cnt']
    self.cur_scale = state['cur_scale']
    self.last_scale = state['last_scale']
