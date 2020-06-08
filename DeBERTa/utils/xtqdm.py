
from tqdm import tqdm
import os

__all__=['xtqdm']

class dummy_tqdm():
  def __init__(self, iterable=None, *wargs, **kwargs):
    self.iterable = iterable

  def __iter__(self):
    for d in self.iterable:
      yield d

  def update(self, *wargs, **kwargs):
    pass

  def close(self):
    pass

def xtqdm(iterable=None, *wargs, **kwargs):
  disable = False
  if 'disable' in kwargs:
    disable = kwargs['disable']
  if 'NO_TQDM' in os.environ:
    disable = True if os.getenv('NO_TQDM', '0')!='0' else False
  if disable:
    return dummy_tqdm(iterable, *wargs, **kwargs)
  else:
    return tqdm(iterable, *wargs, **kwargs)
