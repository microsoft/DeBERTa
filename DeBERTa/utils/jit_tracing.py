"""
Logging util
@Author: penhe@microsoft.com
"""

""" Utils for torch jit tracing customer operators/functions
"""
import os

def traceable(cls):
  """ Decorator over customer functions
      There is an issue for tracing customer python torch Function, using this decorator to work around it.
      e.g.
      @traceable
      class MyOp(torch.autograd.Function):
      xxx
  """

  class _Function(object):
    @staticmethod
    def apply(*args):
      jit_trace = (os.getenv('JIT_TRACE', 'False').lower() == 'true')
      if jit_trace:
        return cls.forward(_Function, *args)
      else:
        return cls.apply(*args)

    @staticmethod
    def save_for_backward(*args):
      pass

  _Function.__name__ = cls.__name__
  _Function.__doc__ = cls.__doc__
  return _Function

class TraceMode():
  """ Trace context used when tracing modules contains customer operators/Functions
  """
  def __enter__(self):
    os.environ['JIT_TRACE'] = 'True'
    return self

  def __exit__(self, exp_value, exp_type, trace):
    del os.environ['JIT_TRACE']
