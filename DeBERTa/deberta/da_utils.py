import torch
import pdb
from functools import lru_cache
import numpy as np

__all__=['build_relative_position', 'make_log_bucket_position']

def make_log_bucket_position(relative_pos, bucket_size, max_position):
  sign = np.sign(relative_pos)
  mid = bucket_size//2
  abs_pos = np.where((relative_pos<mid) & (relative_pos > -mid), mid-1, np.abs(relative_pos))
  log_pos = np.ceil(np.log(abs_pos/mid)/np.log((max_position-1)/mid) * (mid-1)) + mid
  bucket_pos = np.where(abs_pos<=mid, relative_pos, log_pos*sign).astype(np.int)
  return bucket_pos

@lru_cache(maxsize=128)
def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
  q_ids = np.arange(0, query_size)
  k_ids = np.arange(0, key_size)
  rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0],1))
  if bucket_size>0 and max_position > 0:
    rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
  rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
  rel_pos_ids = rel_pos_ids[:query_size, :]
  rel_pos_ids = rel_pos_ids.unsqueeze(0)
  return rel_pos_ids

def test_log_bucket():
  x=np.arange(-511,511)
  y=make_log_bucket_position(x, 128, 512)
  pdb.set_trace()


