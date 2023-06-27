# coding: utf-8
from DeBERTa import deberta
import sys
import argparse
from tqdm import tqdm

def tokenize_data(input, output=None, max_seq_length=512):
  p,t=deberta.load_vocab(vocab_path=None, vocab_type='spm', pretrained_id='deberta-v3-base')
  tokenizer=deberta.tokenizers[t](p)
  if output is None:
    output=input + '.spm'
  all_tokens = []
  lines = 0
  wfs = open(output, 'w', encoding = 'utf-8')
  with open(input, encoding='utf-8') as fs:
    for l in tqdm(fs, ncols=80, desc='processing...'):
      if len(l) > 0:
        tokens = tokenizer.tokenize(l)
      else:
        tokens = []
      all_tokens.extend(tokens)
      if len(all_tokens) >= max_seq_length-2:
        wfs.write(' '.join(all_tokens[:max_seq_length-2]) + '\n')
        all_tokens = all_tokens[max_seq_length-2:]
        lines += 1
  wfs.close()
  print(f'Saved {lines} lines to {output}')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='The input data path')
parser.add_argument('-o', '--output', default=None, help='The output data path')
parser.add_argument('--max_seq_length', type=int, default=512, help='Maxium sequence length of inputs')
args = parser.parse_args()
tokenize_data(args.input, args.output, args.max_seq_length)
