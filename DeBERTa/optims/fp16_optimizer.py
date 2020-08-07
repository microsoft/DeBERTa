# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 05/30/2019
#
""" FP16 optimizer wrapper
"""

from collections import defaultdict
import numpy as np
import math
import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import ctypes

from ..utils import get_logger
logger=get_logger()

try:
  lib = ctypes.cdll.LoadLibrary(None)
  lib.THCudaHalfTensor_normall.argtypes=[ctypes.c_void_p, ctypes.c_void_p]
  lib.THCudaHalfTensor_normall.restype = ctypes.c_float
except:
  lib = None
  logger.warning('Failed to load half normal.')
  pass


__all__ = ['Fp16Optimizer', 'ExpLossScaler']

def get_world_size():
  try:
    wd = dist.get_world_size()
    return wd
  except:
    return 1

def fused_norm(input):
  if input.type() == 'torch.cuda.HalfTensor':
    if (lib is not None):
      return lib.THCudaHalfTensor_normall(torch.cuda._state_cdata, input._cdata, 16384)
    else:
      return input.norm()
  else:
    return input.norm()

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
  def __init__(self, param_groups, optimizer_fn, loss_scaler=None, grad_clip_norm = 1.0, lookahead_k = -1, lookahead_alpha = 0.5):
    # all parameters should on the same device
    groups = []
    original_groups = []
    for group in param_groups:
      params = group['params'] # parameter
      flattened_params = _flatten_dense_tensors([p.data for p in params])
      unflattend_params = _unflatten_dense_tensors(flattened_params, [p.data for p in params])
      for uf,p in zip(unflattend_params, params):
        p.data = uf
      
      if params[0].dtype==torch.half:
        master_params = flattened_params.clone().to(torch.float).detach_().to(flattened_params.device)
        group['params'] = [OptParameter(master_params, flattened_params, name='master')]
      else:
        group['params'] = [OptParameter(flattened_params, None, name='master')]
      
      o_group = defaultdict(list)
      o_group['names'] = group['names']
      o_group['params'] = params

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

  def get_master_params(self):
    params = []
    for g, o_g in zip(self.param_groups, self.original_param_groups):
      ops = [p.data for p in o_g['params']]
      unflattend_params = _unflatten_dense_tensors(g['params'][0].data, ops)
      for up, op, n in zip(unflattend_params, o_g['params'], o_g['names']):
        params.append((n, torch.nn.Parameter(up)))
    
    state = defaultdict(dict)
    state['params'] = params
    return state

  def get_params_norm(self):
    norm0 = torch.zeros(1, dtype=torch.float)
    norm1 = torch.zeros(1, dtype=torch.float)
    for g in self.param_groups:
      for p in g['params']:
        norm0  = norm0.to(p.data.device)
        norm1  = norm1.to(p.data.device)
        norm0 += fused_norm(p.data)
        if p.out_data is not None:
          norm1 += fused_norm(p.out_data)
    norm = torch.cat((norm0, norm1))
    wd = get_world_size()
    if wd > 1:
      norms = [torch.zeros_like(norm) for _ in range(wd)]
      dist.all_gather(norms, norm)
    else:
      norms = [norm]
    return norms

  def state_dict(self):
    state = defaultdict(dict)
    opt_state = self.optimizer.state_dict()
    state['optimizer'] = opt_state
    master_params = [p['params'][0].data for p in self.param_groups if p['params'][0].out_data is not None]
    state['master_params'] = master_params
    if self.loss_scaler is not None:
      state['scaler'] = self.loss_scaler.state_dict()

    return state

  def load_state_dict(self, state):
    opt_state = state['optimizer']
    self.optimizer.load_state_dict(opt_state)
    self.param_groups = self.optimizer.param_groups
    saved_master_params = state['master_params']
    master_params = [p['params'][0].data for p in self.param_groups if p['params'][0].out_data is not None]
    assert len(saved_master_params) == len(master_params), f'Saved master parameters must matches the master parameters in the object.'
    for s, m in zip(saved_master_params, master_params):
      assert s.size()==m.size()
      m.copy_(s)

    if self.loss_scaler is not None:
      self.loss_scaler.load_state_dict(state['scaler'])

  def _grad_scale(self, loss_scale = 1):
    norm = torch.zeros(1, dtype=torch.float)
    for group, o_g in zip(self.param_groups, self.original_param_groups):
      grads = [p.grad if p.grad is not None else torch.zeros_like(p.data) for p in o_g['params']]

      flattened_grads = _flatten_dense_tensors(grads)
      wd = get_world_size()
      if wd > 1:
        loss_scale *= wd
        dist.all_reduce(flattened_grads)
        torch.cuda.synchronize()

      norm = norm.to(flattened_grads.device)
      norm = norm + fused_norm(flattened_grads)**2
      group['params'][0].grad = flattened_grads
    norm = norm**0.5
    if torch.isnan(norm) or torch.isinf(norm) :
      return None

    grad_scale = loss_scale
    if self.max_grad_norm>0:
      scale = norm/(loss_scale*self.max_grad_norm)
      if scale>1:
        grad_scale *= scale
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
