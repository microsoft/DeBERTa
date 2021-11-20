#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Pengcheng He (penhe@microsoft.com)
# Date: 05/15/2019
#

import os
import numpy as np
import math
import sys
from torch.utils.data import Sampler

__all__=['BatchSampler', 'DistributedBatchSampler', 'RandomSampler', 'SequentialSampler']
class BatchSampler(Sampler):
  def __init__(self, sampler, batch_size):
    self.sampler = sampler
    self.batch_size = batch_size

  def __iter__(self):
    batch = []
    for idx in self.sampler:
      batch.append(idx)
      if len(batch)==self.batch_size:
        yield batch
        batch = []
    if len(batch)>0:
      yield batch

  def __len__(self):
    return (len(self.sampler) + self.batch_size - 1)//self.batch_size

class DistributedBatchSampler(Sampler):
  def __init__(self, sampler, rank=0, world_size = 1, drop_last = False):
    self.sampler = sampler
    self.rank = rank
    self.world_size = world_size
    self.drop_last = drop_last

  def __iter__(self):
    for b in self.sampler:
      if len(b)%self.world_size != 0:
        if self.drop_last:
          break
        else:
          b.extend([b[0] for _ in range(self.world_size-len(b)%self.world_size)])
      chunk_size = len(b)//self.world_size
      yield b[self.rank*chunk_size:(self.rank+1)*chunk_size]

  def __len__(self):
    return len(self.sampler)

class RandomSampler(Sampler):
  def __init__(self, total_samples:int, data_seed:int = 0):
    self.indices = np.array(np.arange(total_samples))
    self.rng = np.random.RandomState(data_seed)

  def __iter__(self):
    self.rng.shuffle(self.indices)
    for i in self.indices:
      yield i

  def __len__(self):
    return len(self.indices)

class SequentialSampler(Sampler):
  def __init__(self, total_samples:int):
    self.indices = np.array(np.arange(total_samples))

  def __iter__(self):
    for i in self.indices:
      yield i

  def __len__(self):
    return len(self.indices)
