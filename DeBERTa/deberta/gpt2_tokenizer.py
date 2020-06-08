# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/15/2020
#

# This piece of code is derived from https://github.com/pytorch/fairseq/blob/master/fairseq/data/encoders/gpt2_bpe.py

import torch
import unicodedata
import os
from .gpt2_bpe_utils import get_encoder,_is_control,_is_whitespace,_is_punctuation
from .cache_utils import load_vocab

__all__ = ['GPT2Tokenizer']

class GPT2Tokenizer(object):
  def __init__(self, vocab_file=None, do_lower_case=True, special_tokens=None):
    pad='[PAD]'
    eos='[SEP]'
    unk='[UNK]'
    bos='[CLS]'

    self.symbols = []
    self.count = []
    self.indices = {}
    self.add_symbol(pad)
    self.add_symbol(bos)
    self.add_symbol(eos)
    self.add_symbol(unk)

    gpt2_encoder = load_vocab(vocab_file)
    self.bpe = get_encoder(gpt2_encoder['encoder'], gpt2_encoder['vocab'])
    for w,n in gpt2_encoder['dict_map']:
      self.add_symbol(w, n)

    self.mask_id = self.add_symbol('[MASK]')
    self.special_tokens = ['[MASK]', '[SEP]', '[PAD]', '[UNK]', '[CLS]']
    if special_tokens is not None:
      for t in special_tokens:
        self.add_special_token(t)

    self.vocab = self.indices
    self.ids_to_tokens = self.symbols

  def tokenize(self, text):
    bpe = self._encode(text)

    return [t for t in bpe.split(' ') if t]

  def convert_tokens_to_ids(self, tokens):
    return [self.vocab[t] for t in tokens]

  def convert_ids_to_tokens(self, ids):
    tokens = []
    for i in ids:
      tokens.append(self.ids_to_tokens[i])
    return tokens

  def split_to_words(self, text):
    return self.bpe.split_to_words(text)

  def decode(self, tokens):
    return self.bpe.decode([int(t) for t in tokens if t not in self.special_tokens])

  def add_special_token(self, token):
    self.special_tokens.append(token)
    return self.add_symbol(token)

  def part_of_whole_word(self, token, is_bos=False):
    if is_bos:
      return True
    s = self._decode(token)
    if (len(s)==1 and (_is_whitespace(list(s)[0]) or _is_control(list(s)[0]) or _is_punctuation(list(s)[0]))):
      return False

    return not s.startswith(' ')

  def sym(self, id):
    return self.ids_to_tokens[id]

  def id(self, sym):
    return self.vocab[sym]

  def _encode(self, x: str) -> str:
    return ' '.join(map(str, self.bpe.encode(x)))

  def _decode(self, x: str) -> str:
    return self.bpe.decode(map(int, x.split()))

  def add_symbol(self, word, n=1):
    """Adds a word to the dictionary"""
    if word in self.indices:
      idx = self.indices[word]
      self.count[idx] = self.count[idx] + n
      return idx
    else:
      idx = len(self.symbols)
      self.indices[word] = idx
      self.symbols.append(word)
      self.count.append(n)
      return idx

