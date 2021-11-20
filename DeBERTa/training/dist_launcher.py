#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Pengcheng He (penhe@microsoft.com)
# Date: 05/15/2019
#

import os
import time
import pdb
import signal
import torch
from multiprocessing import Process,Pool
from collections import defaultdict
import sys
import psutil
from ..utils import set_logger, get_logger
logger = get_logger()

def kill_children(proc=None, recursive = True):
  if proc is None:
    proc = psutil.Process()
  _children = proc.children(recursive=False)
  for c in _children:
    try:
      if recursive:
        kill_children(c, recursive=recursive)
      os.kill(c.pid, signal.SIGKILL)
    except:
      pass

  for c in _children:
    try:
      c.wait(1)
    except:
      pass

def gc(i):
  return torch.cuda.device_count()

def get_ngpu():
  with Pool(1) as p:
    return p.map(gc, range(1))[0]

def _setup_distributed_group(args):
  """Initialize torch.distributed."""

  torch.backends.cudnn.enabled = False
  if args.world_size == 1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    set_logger(args.task_name, os.path.join(args.output_dir, f'training_{args.task_name}_{args.rank}.log'), rank=args.rank, verbose=1 if args.local_rank==0 else 0)
    device_id = args.rank % args.n_gpu
    if args.local_rank >= 0:
      device_id = args.local_rank
    device = torch.device("cuda", device_id)
    init_method = 'tcp://'
    init_method += args.master_ip + ':' + args.master_port
    distributed_backend = getattr(args, 'distributed_backend', 'nccl')
    torch.distributed.init_process_group(
      backend=distributed_backend,
      world_size=args.world_size, rank=args.rank,
      init_method=init_method)
    torch.cuda.set_device(device)
  n_gpu = torch.cuda.device_count()
  logger.info("device=%s, n_gpu=%d, distributed training=%r, world_size=%d", device, n_gpu, bool(args.world_size != 1), args.world_size)
  return device

def _get_world_size(args):
    world_size = int(os.getenv("WORLD_SIZE", '1'))
    if not hasattr(args, 'n_gpu') or args.n_gpu is None:
      n_gpu = get_ngpu()
    return n_gpu * world_size

def initialize_distributed(args, join=True):
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    args.rank = int(os.getenv('RANK', '0'))
    args.master_ip = os.getenv('MASTER_ADDR', 'localhost')
    args.master_port = os.getenv('MASTER_PORT', '17006')
  
    if args.world_size == 1:
      args.rank = 0
      args.master_ip = 'localhost'

    if not hasattr(args, 'n_gpu') or args.n_gpu is None:
      args.n_gpu = get_ngpu()

    args.node_rank = args.rank
    args.world_size = args.n_gpu * args.world_size
    seed = args.seed
    is_child = False
    if args.world_size>1:
      children = []
      for r in range(args.n_gpu):
        args.rank = r + args.n_gpu*args.node_rank
        args.local_rank = r
        args.seed = seed + args.rank
        child = os.fork()
        if child>0:
          children.append(child)
        else:
          signal.signal(signal.SIGINT, signal.SIG_IGN)
          is_child = True
          break
    else:
      is_child = True

    if is_child:
      return _setup_distributed_group(args)
    else:
      if join:
        try:
          for c in children:
            cid, ccode = os.waitpid(0,0)
            logger.debug(f'Worker {c} done with code {ccode}')
            if ccode != 0:
              logger.error(f'Worker {c} : {cid} failed with code {ccode}')
              kill_children()
              raise ValueError(f'Job failed. {cid}:{ccode}')
        except (KeyboardInterrupt, SystemExit):
          logger.warning('Keybord interrupt by user. Terminate all processes')
          kill_children(None)
      return children

def test_dist_launch():
  def test_functions(args):
    global logger
    set_logger(args.task_name, os.path.join(args.output_dir, f'training_{args.task_name}_{args.node_rank}.log'), rank=args.rank)
    logger.info(args)

  class Args:
    def __init__(self):
      pass
    def __repr__(self):
      return str(self.__dict__)

  args = Args()
  args.task_name = 'test'
  args.seed = 0
  args.n_gpu = None
  args.no_cuda=False
  args.output_dir = '/tmp'
  distributed_launch(args, test_functions, (args,))

def test_init_dist():
  class Args:
    def __init__(self):
      pass
    def __repr__(self):
      return str(self.__dict__)

  args = Args()
  args.task_name = 'test'
  args.seed = 0
  args.n_gpu = None
  args.no_cuda=False
  args.output_dir = '/tmp'
  device = initialize_distributed(args)
  if isinstance(device, torch.device):
    return 0
  else:
    return 1
