"""
Logging util
@Author: penhe@microsoft.com
"""

__all__ = ['get_logger', 'set_logger']
import logging
import os
import pdb

logging.basicConfig(format = '%(asctime)s|%(levelname)s|%(name)s| %(message)s',
                    datefmt = '%m%d%Y %H:%M:%S',
                    level = logging.INFO)
logger=None
def set_logger(name, file_log=None, rank=0, verbose=1):
    global logger
    if not logger:
      logger = logging.getLogger(name)
    else:
      logger.name = name
    
    dirty_handlers = [h for h in logger.handlers]

    if rank >= 0:
      formatter = logging.Formatter(f'%(asctime)s|%(levelname)s|%(name)s|{rank:02}| %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    else:
      formatter = logging.Formatter(f'%(asctime)s|%(levelname)s|%(name)s| %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    if file_log:
        fh = logging.FileHandler(file_log)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Stdout
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    if verbose > 0:
      ch.setLevel(logging.INFO)
    else:
      ch.setLevel(logging.WARN)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    for h in dirty_handlers:
      logger.removeHandler(h)
    logger.propagate=False
    return logger

def get_logger(name='logging', file_log=None, rank=0, verbose=1):
  global logger
  if not logger:
    logger = set_logger(name, file_log, rank, verbose)
  return logger

