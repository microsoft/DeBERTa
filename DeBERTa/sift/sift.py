# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/25/2021
#

import torch
import torch.nn.functional as F

__all__ = ['PerturbationLayer', 'AdversarialLearner', 'hook_sift_layer']

class PerturbationLayer(torch.nn.Module):
  def __init__(self, hidden_size, learning_rate=1e-4, init_perturbation=1e-2):
    super().__init__()
    self.learning_rate = learning_rate
    self.init_perturbation = init_perturbation
    self.delta = None
    self.LayerNorm = torch.nn.LayerNorm(hidden_size, 1e-7, elementwise_affine=False)
    self.adversarial_mode = False

  def adversarial_(self, adversarial = True):
    self.adversarial_mode = adversarial
    if not adversarial:
      self.delta = None

  def forward(self, input):
    if not self.adversarial_mode:
      self.input = self.LayerNorm(input)
      return self.input
    else:
      if self.delta is None:
        self.update_delta(requires_grad=True)
      return self.perturbated_input

  def update_delta(self, requires_grad = False):
    if not self.adversarial_mode:
      return True
    if self.delta is None:
      delta = torch.clamp(self.input.new(self.input.size()).normal_(0, self.init_perturbation).float(), -2*self.init_perturbation, 2*self.init_perturbation)
    else:
      grad = self.delta.grad
      self.delta.grad = None
      delta = self.delta
      norm = grad.norm()
      if torch.isnan(norm) or torch.isinf(norm):
        return False
      eps = self.learning_rate
      with torch.no_grad():
        delta = delta + eps*grad/(1e-6 + grad.abs().max(-1, keepdim=True)[0])
    self.delta = delta.float().detach().requires_grad_(requires_grad)
    self.perturbated_input = (self.input.to(delta).detach() + self.delta).to(self.input)
    return True

def hook_sift_layer(model, hidden_size, learning_rate=1e-4, init_perturbation=1e-2, target_module = 'embeddings.LayerNorm'):
  """
  Hook the sift perturbation layer to and existing model. With this method, you can apply adversarial training
  without changing the existing model implementation.

  Params:
    `model`: The model instance to apply adversarial training
    `hidden_size`: The dimmension size of the perturbated embedding
    `learning_rate`: The learning rate to update the perturbation
    `init_perturbation`: The initial range of perturbation
    `target_module`: The module to apply perturbation. It can be the name of the sub-module of the model or the sub-module instance.
    The perturbation layer will be inserted before the sub-module.

  Outputs:
    The perturbation layers.

  """
  
  if isinstance(target_module, str):
    _modules = [k for n,k in model.named_modules() if  target_module in n]
  else:
    assert isinstance(target_module, torch.nn.Module), f'{type(target_module)} is not an instance of torch.nn.Module'
    _modules = [target_module]
  adv_modules = []
  for m in _modules:
    adv = PerturbationLayer(hidden_size, learning_rate, init_perturbation)
    def adv_hook(module, inputs):
      return adv(inputs[0])
    for h in list(m._forward_pre_hooks.keys()):
      if m._forward_pre_hooks[h].__name__ == 'adv_hook':
        del m._forward_pre_hooks[h]
    m.register_forward_pre_hook(adv_hook)
    adv_modules.append(adv)
  return adv_modules

class AdversarialLearner:
  """ Adversarial Learner
  This class is the helper class for adversarial training.

  Params:
    `model`: The model instance to apply adversarial training
    `perturbation_modules`: The sub modules in the model that will generate perturbations. If it's `None`,
    the constructor will detect sub-modules of type `PerturbationLayer` in the model.

  Example usage:
  ```python
  # Create DeBERTa model
  adv_modules = hook_sift_layer(model, hidden_size=768)
  adv = AdversarialLearner(model, adv_modules)
  def logits_fn(model, *wargs, **kwargs):
    logits,_ = model(*wargs, **kwargs)
    return logits
  logits,loss = model(**data)

  loss = loss + adv.loss(logits, logits_fn, **data)
  # Other steps is the same as general training.

  ```

  """
  def __init__(self, model, adv_modules=None):
    if adv_modules is None:
      self.adv_modules = [m for m in model.modules() if isinstance(m, PerturbationLayer)]
    else:
      self.adv_modules = adv_modules
    self.parameters = [p for p in model.parameters()]
    self.model = model

  def loss(self, target, logits_fn, loss_fn = 'symmetric-kl', *wargs, **kwargs):
    """
    Calculate the adversarial loss based on the given logits fucntion and loss function.
    Inputs:
    `target`: the logits from original inputs.
    `logits_fn`: the function that produces logits based on perturbated inputs. E.g.,
    ```python
    def logits_fn(model, *wargs, **kwargs):
      logits = model(*wargs, **kwargs)
      return logits
    ```
    `loss_fn`: the function that caclulate the loss from perturbated logits and target logits.
      - If it's a string, it can be pre-built loss functions, i.e. kl, symmetric_kl, mse.
      - If it's a function, it will be called to calculate the loss, the signature of the function will be,
      ```python
      def loss_fn(source_logits, target_logits):
        # Calculate the loss
        return loss
      ```
    `*wargs`: the positional arguments that will be passed to the model
    `**kwargs`: the key-word arguments that will be passed to the model
    Outputs:
      The loss based on pertubated inputs.
    """
    self.prepare()
    if isinstance(loss_fn, str):
      loss_fn = perturbation_loss_fns[loss_fn]
    pert_logits = logits_fn(self.model, *wargs, **kwargs)
    pert_loss = loss_fn(pert_logits, target.detach()).sum()
    pert_loss.backward()
    for m in self.adv_modules:
      ok = m.update_delta(True)

    for r,p in zip(self.prev, self.parameters):
      p.requires_grad_(r)
    pert_logits = logits_fn(self.model, *wargs, **kwargs)
    pert_loss = symmetric_kl(pert_logits, target)

    self.cleanup()
    return pert_loss.mean()

  def prepare(self):
    self.prev = [p.requires_grad for p in self.parameters]
    for p in self.parameters:
      p.requires_grad_(False)
    for m in self.adv_modules:
      m.adversarial_(True)
  
  def cleanup(self):
    for r,p in zip(self.prev, self.parameters):
      p.requires_grad_(r)

    for m in self.adv_modules:
      m.adversarial_(False)

def symmetric_kl(logits, target):
  logit_stu = logits.view(-1, logits.size(-1)).float()
  logit_tea = target.view(-1, target.size(-1)).float()
  logprob_stu = F.log_softmax(logit_stu, -1)
  logprob_tea = F.log_softmax(logit_tea, -1)
  prob_tea = logprob_tea.exp().detach()
  prob_stu = logprob_stu.exp().detach()
  floss = ((prob_tea*(-logprob_stu)).sum(-1))    # Cross Entropy
  bloss = ((prob_stu*(-logprob_tea)).sum(-1))    # Cross Entropy
  loss = floss + bloss
  return loss

def kl(logits, target):
  logit_stu = logits.view(-1, logits.size(-1)).float()
  logit_tea = target.view(-1, target.size(-1)).float()
  logprob_stu = F.log_softmax(logit_stu, -1)
  logprob_tea = F.log_softmax(logit_tea.detach(), -1)
  prob_tea = logprob_tea.exp()
  loss = ((prob_tea*(-logprob_stu)).sum(-1))    # Cross Entropy
  return loss

def mse(logits, target):
  logit_stu = logits.view(-1, logits.size(-1)).float()
  logit_tea = target.view(-1, target.size(-1)).float()
  return F.mse_loss(logit_stu.view(-1),logit_tea.view(-1))

perturbation_loss_fns = {
    'symmetric-kl': symmetric_kl,
    'kl': kl,
    'mse': mse
    }
