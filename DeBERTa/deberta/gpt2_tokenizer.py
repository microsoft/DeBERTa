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
  """ A wrapper of GPT2 tokenizer with similar interface as BERT tokenizer

  Args:
    
    vocab_file (:obj:`str`, optional):
      The local path of vocabulary package or the release name of vocabulary in `DeBERTa GitHub releases <https://github.com/microsoft/DeBERTa/releases>`_, \
          e.g. "bpe_encoder", default: `None`. 
          
          If it's `None`, then it will download the vocabulary in the latest release from GitHub. The vocabulary file is a \
          state dictionary with three items, "dict_map", "vocab", "encoder" which correspond to three files used in `RoBERTa`, i.e. `dict.txt`, `vocab.txt` and `encoder.json`. \
          
          The difference between our wrapped GPT2 tokenizer and RoBERTa wrapped tokenizer are,

          - Special tokens, unlike `RoBERTa` which use `<s>`, `</s>` as the `start` token and `end` token of a sentence. We use `[CLS]` and `[SEP]` as the `start` and `end`\
              token of input sentence which is the same as `BERT`.

          - We remapped the token ids in our dictionary with regarding to the new special tokens, `[PAD]` => 0, `[CLS]` => 1, `[SEP]` => 2, `[UNK]` => 3, `[MASK]` => 50264

    do_lower_case (:obj:`bool`, optional):
      Whether to convert inputs to lower case. **Not used in GPT2 tokenizer**.

    special_tokens (:obj:`list`, optional):
      List of special tokens to be added to the end of the vocabulary.


  """
  def __init__(self, vocab_file=None, do_lower_case=True, special_tokens=None):
    self.pad_token='[PAD]'
    self.sep_token='[SEP]'
    self.unk_token='[UNK]'
    self.cls_token='[CLS]'

    self.symbols = []
    self.count = []
    self.indices = {}
    self.pad_token_id = self.add_symbol(self.pad_token)
    self.cls_token_id = self.add_symbol(self.cls_token)
    self.sep_token_id = self.add_symbol(self.sep_token)
    self.unk_token_id = self.add_symbol(self.unk_token)

    self.gpt2_encoder = torch.load(vocab_file)
    self.bpe = get_encoder(self.gpt2_encoder['encoder'], self.gpt2_encoder['vocab'])
    for w,n in self.gpt2_encoder['dict_map']:
      self.add_symbol(w, n)

    self.mask_token='[MASK]'
    self.mask_id = self.add_symbol(self.mask_token)
    self.special_tokens = ['[MASK]', '[SEP]', '[PAD]', '[UNK]', '[CLS]']
    if special_tokens is not None:
      for t in special_tokens:
        self.add_special_token(t)

    self.vocab = self.indices
    self.ids_to_tokens = self.symbols

  def tokenize(self, text):
    """ Convert an input text to tokens.
      
      Args:
        
        text (:obj:`str`): input text to be tokenized.

      Returns:
        A list of byte tokens where each token represent the byte id in GPT2 byte dictionary

      Example::
        
        >>> tokenizer = GPT2Tokenizer()
        >>> text = "Hello world!"
        >>> tokens = tokenizer.tokenize(text)
        >>> print(tokens)
        ['15496', '995', '0']
        
    """
    bpe = self._encode(text)

    return [t for t in bpe.split(' ') if t]

  def convert_tokens_to_ids(self, tokens):
    """ Convert list of tokens to ids.
      
      Args:

        tokens (:obj:`list<str>`): list of tokens

      Returns:
        
        List of ids
    """

    return [self.vocab[t] for t in tokens]

  def convert_ids_to_tokens(self, ids):
    """ Convert list of ids to tokens.
      
      Args:

        ids (:obj:`list<int>`): list of ids

      Returns:
        
        List of tokens
    """

    tokens = []
    for i in ids:
      tokens.append(self.ids_to_tokens[i])
    return tokens

  def split_to_words(self, text):
    return self.bpe.split_to_words(text)

  def decode(self, tokens):
    """ Decode list of tokens to text strings.
    
      Args:
        
        tokens (:obj:`list<str>`): list of tokens.

      Returns:
        
        Text string corresponds to the input tokens.

      Example::
        
        >>> tokenizer = GPT2Tokenizer()
        >>> text = "Hello world!"
        >>> tokens = tokenizer.tokenize(text)
        >>> print(tokens)
        ['15496', '995', '0']
        
        >>> tokenizer.decode(tokens)
        'Hello world!'
      
    """
    return self.bpe.decode([int(t) for t in tokens if t not in self.special_tokens])

  def add_special_token(self, token):
    """Adds a special token to the dictionary.
    
      Args:
        token (:obj:`str`): Tthe new token/word to be added to the vocabulary.

      Returns:
        The id of new token in the vocabulary.

    """
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
    """Adds a word to the dictionary.
    
      Args:
        word (:obj:`str`): Tthe new token/word to be added to the vocabulary.
        n (int, optional): The frequency of the word.

      Returns:
        The id of the new word.

    """
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

  def save_pretrained(self, path: str):
    torch.save(self.gpt2_encoder, path)
