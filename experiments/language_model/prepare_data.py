# coding: utf-8
from DeBERTa import deberta
import sys
import argparse
from tqdm import tqdm

def tokenize_data(input, output=None):
  p,t=deberta.load_vocab(vocab_path=None, vocab_type='gpm', pretrained_id='xlarge-v2')
  tokenizer=deberta.tokenizers[t](p)
  if output is None:
    output=input + '.gpm'
  all_tokens = []
  with open(input, encoding = 'utf-8') as fs:
    for l in tqdm(fs, ncols=80, desc='Loading'):
      if len(l) > 0:
        tokens = tokenizer.tokenize(l)
      else:
        tokens = []
      all_tokens.extend(tokens)

  print(f'Loaded {len(all_tokens)} tokens from {input}')
  lines = 0
  with open(output, 'w', encoding = 'utf-8') as wfs:
    idx = 0
    while idx < len(all_tokens):
      wfs.write(' '.join(all_tokens[idx:idx+510]) + '\n')
      idx += 510
      lines += 1

  print(f'Saved {lines} lines to {output}')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='The input data path')
parser.add_argument('-o', '--output', default=None, help='The output data path')
args = parser.parse_args()
tokenize_data(args.input, args.output)
