import torch
import pdb
from functools import lru_cache
import numpy as np
import math

__all__=['build_relative_position', 'make_log_bucket_position']

@lru_cache(maxsize=128)
def make_log_bucket_dict(bucket_size, max_position, device=None):
  relative_pos = torch.arange(-max_position, max_position, device=device)
  sign = torch.sign(relative_pos)
  mid = bucket_size//2
  abs_pos = torch.where((relative_pos<mid) & (relative_pos > -mid), torch.tensor(mid-1).to(relative_pos), torch.abs(relative_pos))
  log_pos = torch.ceil(torch.log(abs_pos/mid)/math.log((max_position-1)/mid) * (mid-1)) + mid
  bucket_pos = torch.where(abs_pos<=mid, relative_pos, (log_pos*sign).to(relative_pos)).to(torch.long)
  return bucket_pos

# Faster version
def make_log_bucket_position(relative_pos, bucket_size, max_position):
  relative_pos = torch.clamp(relative_pos,-max_position+1, max_position-1) + max_position
  bucket_dict = make_log_bucket_dict(bucket_size, max_position, relative_pos.device)
  for d in range(relative_pos.dim()-1):
    bucket_dict = bucket_dict.unsqueeze(0)
    bucket_pos = torch.gather(bucket_dict.expand(list(relative_pos.size())[:-1] + [bucket_dict.size(-1)]), index=relative_pos.long(), dim=-1)
  return bucket_pos

@lru_cache(maxsize=128)
def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1, device=None):
  q_ids = torch.arange(0, query_size)
  k_ids = torch.arange(0, key_size)
  if device is not None:
    q_ids = q_ids.to(device)
    k_ids = k_ids.to(device)
  rel_pos_ids = q_ids.view(-1,1) - k_ids.view(1,-1)
  #q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0],1))
  if bucket_size>0 and max_position > 0:
    rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
  #rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
  rel_pos_ids = rel_pos_ids[:query_size, :]
  rel_pos_ids = rel_pos_ids.unsqueeze(0)
  return rel_pos_ids

def build_relative_position_from_abs(query_pos, key_pos, bucket_size=-1, max_position=-1, device=None):
  if isinstance(query_pos, tuple):
    q_ids = torch.tensor(query_pos)
  else:
    q_ids = query_pos
  if isinstance(key_pos, tuple):
    k_ids = torch.tensor(key_pos)
  else:
    k_ids = key_pos

  if device is not None:
    q_ids = q_ids.to(device)
    k_ids = k_ids.to(device)
  rel_pos_ids = q_ids.unsqueeze(-1) - k_ids.unsqueeze(-2)
  #q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0],1))
  if bucket_size>0 and max_position > 0:
    rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
  #rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
  return rel_pos_ids

def test_log_bucket():
  x=np.arange(-511,511)
  y=make_log_bucket_position(x, 128, 512)
  pdb.set_trace()

