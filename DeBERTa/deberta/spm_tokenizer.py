# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 11/15/2020
#


import sentencepiece as sp
import six
import unicodedata
import os
import regex as re
from .cache_utils import load_vocab
from ..utils import get_logger
logger=get_logger()


import pdb

__all__ = ['SPMTokenizer']

class SPMTokenizer:
  def __init__(self, vocab_file, do_lower_case=False, special_tokens=None, bpe_dropout=0, split_by_punct=False):
    self.split_by_punct = split_by_punct
    spm = sp.SentencePieceProcessor()
    assert os.path.exists(vocab_file)
    spm.load(vocab_file)
    bpe_vocab_size = spm.GetPieceSize()
    # Token map
    # <unk> 0+1
    # <s> 1+1
    # </s> 2+1
    self.vocab = {spm.IdToPiece(i):i for i in range(bpe_vocab_size)}
    self.id_to_tokens = [spm.IdToPiece(i) for i in range(bpe_vocab_size)]
    #self.vocab['[PAD]'] = 0
    #self.vocab['[CLS]'] = 1
    #self.vocab['[SEP]'] = 2
    #self.vocab['[UNK]'] = 3

    _special_tokens = ['[MASK]', '[SEP]', '[PAD]', '[UNK]', '[CLS]']
    self.special_tokens = []
    if special_tokens is not None:
      _special_tokens.extend(special_tokens)
    for t in _special_tokens:
      self.add_special_token(t)

    self.spm = spm
    self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

  def tokenize(self, text):
    pieces = self._encode_as_pieces(text)
    def _norm(x):
      if x not in self.vocab or x=='<unk>':
        return '[UNK]'
      else:
        return x
    pieces = [_norm(p) for p in pieces]
    return pieces

  def convert_tokens_to_ids(self, tokens):
    return [self.vocab[t] if t in self.vocab else 1 for t in tokens]

  def convert_ids_to_tokens(self, ids):
    tokens = []
    for i in ids:
      tokens.append(self.ids_to_tokens[i])
    return tokens

  def decode(self, tokens, start=-1, end=-1, raw_text=None):
    if raw_text is None:
      return self.spm.decode_pieces([t for t in tokens if t not in self.special_tokens])
    else:
      words = self.split_to_words(raw_text)
      word_tokens = [self.tokenize(w) for w in words]
      wt = [w for t in word_tokens for w in t]
      #assert tokens == wt, f'{tokens} || {wt}'
      if wt!=tokens:
        for a,b in zip(wt, tokens):
          if a!=b:
            pdb.set_trace()
      token2words = [0]*len(tokens)
      tid = 0
      for i,w in enumerate(word_tokens):
        for k,t in enumerate(w):
          token2words[tid] = i
          tid += 1
      word_start = token2words[start]
      word_end = token2words[end] if end <len(tokens) else len(words)
      text = ''.join(words[word_start:word_end])
      return text

  def add_special_token(self, token):
    if token not in self.special_tokens:
      self.special_tokens.append(token)
      if token not in self.vocab:
        self.vocab[token] = len(self.vocab)
        self.id_to_tokens.append(token)
    return self.id(token)

  def part_of_whole_word(self, token, is_bos=False):
    if is_bos:
      return True
    if (len(token)==1 and (_is_whitespace(list(token)[0]) or _is_control(list(token)[0]) or _is_punctuation(list(token)[0]))) or token in self.special_tokens:
      return False

    word_start = b'\xe2\x96\x81'.decode('utf-8')
    return not token.startswith(word_start)

  def pad(self):
    return '[PAD]'

  def bos(self):
    return '[CLS]'

  def eos(self):
    return '[SEP]'

  def unk(self):
      return '[UNK]'

  def mask(self):
      return '[MASK]'

  def sym(self, id):
    return self.ids_to_tokens[id]

  def id(self, sym):
    return self.vocab[sym] if sym in self.vocab else 1

  def _encode_as_pieces(self, text):
    text = convert_to_unicode(text)
    if self.split_by_punct:
      words = self._run_split_on_punc(text)
      pieces = [self.spm.encode_as_pieces(w) for w in words]
      return [p for w in pieces for p in w]
    else:
      return self.spm.encode_as_pieces(text)

  def split_to_words(self, text):
    pieces = self._encode_as_pieces(text)
    word_start = b'\xe2\x96\x81'.decode('utf-8')
    words = []
    offset = 0
    prev_end = 0
    for i,p in enumerate(pieces):
      if p.startswith(word_start):
        if offset>prev_end:
          words.append(text[prev_end:offset])
        prev_end = offset
        w = p.replace(word_start, '')
      else:
        w = p
      try:
        s = text.index(w, offset)
        pn = ""
        k = i+1
        while k < len(pieces):
          pn = pieces[k].replace(word_start, '')
          if len(pn)>0:
            break
          k += 1

        if len(pn)>0 and pn in text[offset:s]:
          offset = offset + 1
        else:
          offset = s + len(w)
      except:
        offset = offset + 1

    if prev_end< offset:
      words.append(text[prev_end:offset])

    return words

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    #words = list(re.findall(self.pat, text))
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]
  
  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
      (cp >= 0x3400 and cp <= 0x4DBF) or  #
      (cp >= 0x20000 and cp <= 0x2A6DF) or  #
      (cp >= 0x2A700 and cp <= 0x2B73F) or  #
      (cp >= 0x2B740 and cp <= 0x2B81F) or  #
      (cp >= 0x2B820 and cp <= 0x2CEAF) or
      (cp >= 0xF900 and cp <= 0xFAFF) or  #
      (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True
  
    return False
  
  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

