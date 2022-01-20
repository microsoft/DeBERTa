#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Pengcheng He (penhe@microsoft.com)
# Date: 05/15/2019
#

import os
import torch
import random
import time
import numpy as np
import pdb
from collections import defaultdict, Mapping, Sequence, OrderedDict
from torch.utils.data import DataLoader
from ..data import BatchSampler, DistributedBatchSampler,RandomSampler,SequentialSampler, AsyncDataLoader
from ..utils import get_logger
logger = get_logger()

from .dist_launcher import get_ngpu
from .optimizer_utils import create_xoptimizer
from ._utils import batch_to

__all__ = ['DistributedTrainer', 'set_random_seed']

def set_random_seed(seed, cpu_only=False):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  n_gpu = get_ngpu()
  if n_gpu > 0 and not cpu_only:
    torch.cuda.manual_seed_all(seed)

class TrainerState:
  def __init__(self, training_steps, name=None):
    self.__dict__ = defaultdict(float)
    self.loss = 0.0
    self.examples = 0
    self.steps = 0
    self._last_report_step = 0
    self.epochs = 0
    self.next_batch = 0
    self.num_training_steps = training_steps
    self._last_report_time = time.time()
    self.best_steps = 0
    self.best_metric = -1e9
    self.name = name
    self.run_id = None

  def update_step(self, loss, examples, loss_scale):
    self.examples += examples
    self.loss += loss
    self.steps += 1
    self.next_batch += 1
    self.loss_scale = loss_scale
  
  def report_state(self):
    if self.steps <= self._last_report_step:
      return

    end = time.time()
    start = self._last_report_time
    if self.name is not None:
      tag = f'[{self.name}]'
    else:
      tag = None
    logger.info('{}[{:0.1f}%][{:0.2f}h] Steps={}, loss={}, examples={}, loss_scale={:0.1f}, {:0.1f}s'.format(tag, 100*self.steps/self.num_training_steps, \
      (self.num_training_steps - self.steps)*(start-end)/((self.steps-self._last_report_step)*3600), self.steps, self.loss/self.steps, self.examples, self.loss_scale, end-start))
    self._last_report_time = end
    self._last_report_step = self.steps

class DistributedTrainer:
  def __init__(self, args, output_dir, model, device, data_fn, loss_fn=None, optimizer_fn=None, eval_fn=None, init_fn=None, update_fn=None, dump_interval = 10000, name=None, **kwargs):
    """
    data_fn return tuples (training_dataset, training_steps, train_sampler, batch_scheduler), training_dataset is required
    loss_fn return the loss of current mini-batch and the size of the batch
    optimizer_fn return the created optimizer
    eval_fn return metrics for model selection
    """
    self.__dict__.update(kwargs)
    self.args = args
    self.device = device
    self.eval_fn = eval_fn
    self.accumulative_update = 1
    if hasattr(args, 'accumulative_update'):
      self.accumulative_update = args.accumulative_update
    
    train_data, training_steps, train_sampler = data_fn(self)
    self.train_data = train_data
    self.train_sampler = train_sampler if train_sampler is not None else RandomSampler(len(train_data))
    self.training_epochs = int(getattr(args, 'num_train_epochs', 1))

    if training_steps is None:
      training_steps = getattr(args, 'training_steps', (len(training_data) + self.args.train_batch_size-1)//self.args.train_batch_size*self.training_epochs)
    self.training_steps = training_steps

    self.output_dir = output_dir
    self.init_fn = init_fn
    self.trainer_state = TrainerState(self.training_steps, name = name)
    self.dump_interval = dump_interval

    self.model = self._setup_model(args, model)
    self.post_loss_fn = None

    def _opt_fn(trainer, model, training_steps):
      return create_xoptimizer(model, args, num_train_steps = training_steps)
    optimizer_fn = optimizer_fn if optimizer_fn is not None else _opt_fn

    self.optimizer = optimizer_fn(self, model, training_steps)

    def _loss_fn(trainer, model, batch):
      _,loss = model(**batch)
      batch_size = batch['input_ids'].size(0)
      return loss.mean(), batch_size
    self.loss_fn = loss_fn if loss_fn is not None else _loss_fn

    self.initialized = False
    self.update_fn = update_fn

  def initialize(self):
    set_random_seed(self.args.seed)

    if self.args.world_size>1:
      torch.distributed.barrier()
    self.initialized = True

  def train(self):
    if not self.initialized:
      self.initialize()

    rank = self.args.rank
    world_size = self.args.world_size
    for n_epoch in range(self.trainer_state.epochs, self.training_epochs):
      batch_sampler = BatchSampler(self.train_sampler, self.args.train_batch_size)
      batch_sampler = DistributedBatchSampler(batch_sampler, rank = rank, world_size = world_size)
      batch_sampler.next = self.trainer_state.next_batch
      num_workers = getattr(self.args, 'workers', 2)
      train_dataloader = DataLoader(self.train_data, batch_sampler=batch_sampler, num_workers=num_workers, worker_init_fn=self.init_fn, pin_memory=False)
      torch.cuda.empty_cache()
      for step, batch in enumerate(AsyncDataLoader(train_dataloader, 100)):
        if self.trainer_state.steps >= self.training_steps:
          break
        bs_scale = 1
        batch = batch_to(batch, self.device)
        self._train_step(batch, bs_scale)
      # Save model
      self.trainer_state.epochs += 1
      self.trainer_state.next_batch = 0
      self.trainer_state.report_state()
      self._eval_model()

  def save_model(self, args, checkpoint_dir, chk_postfix, model, optimizer):
    save_path= os.path.join(checkpoint_dir, f'pytorch.model-{chk_postfix}.bin')
    if hasattr(model, 'module'):
      model_state = OrderedDict([(n,p) for n,p in model.module.state_dict().items()])
    else:
      model_state = OrderedDict([(n,p) for n,p in model.state_dict().items()])
    if args.rank < 1:
      torch.save(model_state, save_path)
    return save_path

  def _eval_model(self, with_checkpoint=True):
    if with_checkpoint:
      checkpoint_dir = getattr(self.args, 'checkpoint_dir', None)
      checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else self.output_dir
      chk_postfix = f'{self.trainer_state.steps:06}'
      self.save_model(self.args, checkpoint_dir, chk_postfix, self.model, self.optimizer)
    _metric = self.trainer_state.best_metric
    _steps = self.trainer_state.best_steps
    if self.eval_fn is not None:
      metric = self.eval_fn(self, self.model, self.device, tag=f'{self.trainer_state.steps:06}-{self.training_steps}')
      if metric > _metric:
        _metric = metric
        _steps = self.trainer_state.steps
      logger.info(f'Best metric: {_metric}@{_steps}')
    self.trainer_state.best_metric, self.trainer_state.best_steps =  _metric, _steps

  def _train_step(self, data, bs_scale):
    self.model.train()
    go_next=False

    def split(batch, parts):
      sub_batches = [{} for _ in range(parts)]
      for k in batch.keys():
        b = batch[k].size(0)
        s = (b + parts - 1)//parts
        v = batch[k].split(s)
        for i,z in enumerate(v):
          sub_batches[i][k]=z
      chunks = [b for b in sub_batches if len(b)>0]
      return chunks

    if self.accumulative_update>1:
      data_chunks = split(data, self.accumulative_update)
    else:
      data_chunks = [data]

    while not go_next:
      step_loss = 0
      batch_size = 0
      self.optimizer.zero_grad()
      forward_outputs = []
      for i, sub in enumerate(data_chunks):
        output = self.loss_fn(self, self.model, sub)
        if isinstance(output, dict):
          loss, sub_size = output['loss'], output['batch_size']
        else:
          loss, sub_size = output
        forward_outputs.append(output)
        loss = loss/len(data_chunks)
        if i == 0:
          loss_scale, _loss = self.optimizer.backward(loss)
        else:
          _loss = loss.float().detach().item()
          loss = loss.float() * loss_scale
          loss.backward()
        step_loss += _loss
        batch_size += sub_size
      if not self.optimizer.step(bs_scale, loss_scale):
        self.optimizer.zero_grad()
        continue
      go_next = True
    self.trainer_state.update_step(step_loss, batch_size , loss_scale)
    if self.update_fn is not None:
      self.update_fn(self, self.model, loss_scale)
    self.optimizer.zero_grad()

    if self.post_loss_fn is not None:
      self.post_loss_fn(forward_outputs)

    if self.trainer_state.steps%100 == 0:
      self.trainer_state.report_state()
    if self.trainer_state.steps%self.dump_interval == 0:
      self._eval_model()

  def _setup_model(self, args, model):
    if args.world_size > 1:
      for p in model.parameters():
        torch.distributed.broadcast(p.data, 0)
      torch.cuda.synchronize()
    return model
