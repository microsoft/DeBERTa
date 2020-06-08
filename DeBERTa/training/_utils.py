import torch
from collections import Sequence, Mapping

def batch_apply(batch, fn):
  if isinstance(batch, torch.Tensor):
    return fn(batch)
  elif isinstance(batch, Sequence):
    return [batch_apply(x, fn) for x in batch]
  elif isinstance(batch, Mapping):
    return {x:batch_apply(batch[x], fn) for x in batch}
  else:
    raise NotImplementedError(f'Type of {type(batch)} are not supported in batch_apply')

def batch_to(batch, device):
  return batch_apply(batch, lambda x: x.to(device))

