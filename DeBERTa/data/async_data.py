#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Pengcheng He (penhe@microsoft.com)
# Date: 05/15/2019
#

from queue import Queue,Empty
from threading import Thread
class AsyncDataLoader(object):
  def __init__(self, dataloader, buffer_size=100):
    self.buffer_size = buffer_size
    self.dataloader = dataloader

  def __iter__(self):
    queue = Queue(self.buffer_size)
    dl=iter(self.dataloader)
    def _worker():
      while True:
        try:
          queue.put(next(dl))
        except StopIteration:
          break
      queue.put(None)
    t=Thread(target=_worker)
    t.start()
    while True:
      d = queue.get()
      if d is None:
        break
      yield d
    del t
    del queue

  def __len__(self):
    return len(self.dataloader)

