# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/15/2020
#

import pdb
import math
from packaging import version
import torch
from torch.nn import LayerNorm
from ..utils.jit_tracing import traceable

if version.Version(torch.__version__) >= version.Version('1.0.0'):
  from torch import _softmax_backward_data as _softmax_backward_data
else:
  from torch import softmax_backward_data as _softmax_backward_data

__all__ = ['StableDropout', 'MaskedLayerNorm', 'XSoftmax', 'ACT2FN', 'LayerNorm']

@traceable
class XSoftmax(torch.autograd.Function):
  """ Masked Softmax which is optimized for saving memory

  Args:
      
    input (:obj:`torch.tensor`): The input tensor that will apply softmax.
    mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax caculation.
    dim (int): The dimenssion that will apply softmax.
    
  Example::

    import torch
    from DeBERTa.deberta import XSoftmax
    # Make a tensor
    x = torch.randn([4,20,100])
    # Create a mask
    mask = (x>0).int()
    y = XSoftmax.apply(x, mask, dim=-1)
      
  """

  @staticmethod
  def forward(self, input, mask, dim):
    """
    """

    self.dim = dim
    if version.Version(torch.__version__) >= version.Version('1.2.0a'):
      rmask = ~(mask.bool())
    else:
      rmask = (1-mask).byte() # This line is not supported by Onnx tracing.

    output = input.masked_fill(rmask, float('-inf'))
    output = torch.softmax(output, self.dim)
    output.masked_fill_(rmask, 0)
    self.save_for_backward(output)
    return output

  @staticmethod
  def backward(self, grad_output):
    """
    """

    output, = self.saved_tensors
    inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
    return inputGrad, None, None

  @staticmethod
  def symbolic(g, self, mask, dim):
      import torch.onnx.symbolic_helper as sym_help
      from torch.onnx.symbolic_opset9 import masked_fill, softmax

      mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx['Long'])
      r_mask = g.op("Cast", g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value), to_i=sym_help.cast_pytorch_to_onnx['Byte'])
      output = masked_fill(g, self, r_mask, g.op("Constant", value_t=torch.tensor(float('-inf'))))
      output = softmax(g, output, dim)
      return masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.uint8)))

class DropoutContext(object):
  def __init__(self):
    self.dropout = 0
    self.mask = None
    self.scale = 1
    self.reuse_mask = True

def get_mask(input, local_context):
  if not isinstance(local_context, DropoutContext):
    dropout = local_context
    mask = None
  else:
    dropout = local_context.dropout
    dropout *= local_context.scale
    mask = local_context.mask if local_context.reuse_mask else None

  if dropout>0 and mask is None:
    if version.Version(torch.__version__) >= version.Version('1.2.0a'):
      mask=(1-torch.empty_like(input).bernoulli_(1-dropout)).bool()
    else:
      mask=(1-torch.empty_like(input).bernoulli_(1-dropout)).byte()
  
  if isinstance(local_context, DropoutContext):
    if local_context.mask is None:
      local_context.mask = mask

  return mask, dropout

@traceable
class XDropout(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, local_ctx):
    mask, dropout = get_mask(input, local_ctx)
    ctx.scale=1.0/(1-dropout)
    if dropout>0:
      ctx.save_for_backward(mask)
      return input.masked_fill(mask, 0)*ctx.scale
    else:
      return input

  @staticmethod
  def backward(ctx, grad_output):
    if ctx.scale > 1:
      mask, = ctx.saved_tensors
      return grad_output.masked_fill(mask, 0)*ctx.scale, None
    else:
      return grad_output, None

class StableDropout(torch.nn.Module):
  """ Optimized dropout module for stabilizing the training

  Args:

    drop_prob (float): the dropout probabilities

  """

  def __init__(self, drop_prob):
    super().__init__()
    self.drop_prob = drop_prob
    self.count = 0
    self.context_stack = None

  def forward(self, x):
    """ Call the module

    Args:
      
      x (:obj:`torch.tensor`): The input tensor to apply dropout


    """
    if self.training and self.drop_prob>0:
      return XDropout.apply(x, self.get_context())
    return x

  def clear_context(self):
    self.count = 0
    self.context_stack = None

  def init_context(self, reuse_mask=True, scale = 1):
    if self.context_stack is None:
      self.context_stack = []
    self.count = 0
    for c in self.context_stack:
      c.reuse_mask = reuse_mask
      c.scale = scale

  def get_context(self):
    if self.context_stack is not None:
      if self.count >= len(self.context_stack):
        self.context_stack.append(DropoutContext())
      ctx = self.context_stack[self.count]
      ctx.dropout = self.drop_prob
      self.count += 1
      return ctx
    else:
      return self.drop_prob

def MaskedLayerNorm(layerNorm, input, mask = None):
  """ Masked LayerNorm which will apply mask over the output of LayerNorm to avoid inaccurate updatings to the LayerNorm module.
  
  Args:
    layernorm (:obj:`~DeBERTa.deberta.LayerNorm`): LayerNorm module or function
    input (:obj:`torch.tensor`): The input tensor
    mask (:obj:`torch.IntTensor`): The mask to applied on the output of LayerNorm where `0` indicate the output of that element will be ignored, i.e. set to `0`

  Example::

    # Create a tensor b x n x d
    x = torch.randn([1,10,100])
    m = torch.tensor([[1,1,1,0,0,0,0,0,0,0]], dtype=torch.int)
    LayerNorm = DeBERTa.deberta.LayerNorm(100)
    y = MaskedLayerNorm(LayerNorm, x, m)

  """
  output = layerNorm(input).to(input)
  if mask is None:
    return output
  if mask.dim()!=input.dim():
    if mask.dim()==4:
      mask=mask.squeeze(1).squeeze(1)
    mask = mask.unsqueeze(2)
  mask = mask.to(output.dtype)
  return output*mask

def gelu(x):
  """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
  """
  return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
  return x * torch.sigmoid(x)

def linear_act(x):
  return x

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish, "tanh": torch.tanh, "linear": linear_act, 'sigmoid': torch.sigmoid}


