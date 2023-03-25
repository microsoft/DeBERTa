# Pre-train an efficient transformer language model for natual language understanding

## Data

We use [wiki103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) data as example, which is publicly available. It contains three text files, `train.txt`, `valid.txt` and `text.txt`. We use `train.txt` to train the model and `valid.txt` to evalute the intermeidate checkpoints. We first need to run `prepara_data.py` to tokenize these text files. We concatenate all documents into a single text and split it into lines of tokens, while each line has at most 510 token (2 tokens are left to special tokens `[CLS]` and `[SEP]`). 

## Pre-training with Masked Language Modeling task

Run  `mlm.sh` to train a bert like model which uses MLM as the pre-training task. For example,

`mlm.sh bert-base` will train a bert base model which uses absolute position encoding

`mlm.sh deberta-base` will train a deberta base model which uses **Disentangled Attention**

## Pre-training with Replaced Token Detection task

Run `rtd.sh` to train a ELECTRA like model using RTD as the pre-training task. For example,

`rtd.sh deberta-v3-xsmall` will train a DeBERTaV3 XSmall model with 9M backbone parameters(12 layers, 256 hidden size) plus 32M embedding parameters(128k vocabulary size)

`rtd.sh deberta-v3-base` will train a DeBERTaV3 Base model with 81M backbone parameters(12 layers, 768 hidden size) plus 96M embedding parameters(128k vocabulary size)

`rtd.sh deberta-v3-large` will train a DeBERTaV3 Large model with 288M backbone parameters(24 layers, 1024 hidden size) plus 128M embedding parameters(128k vocabulary size)

## Continuously training with Replaced Token Detection

Run `rtd.sh deberta-v3-X-continue` to continuously train DeBERTaV3-X models with following checkpoints. 

Please check the script to specify initialization models of generator and discriminator.

|Model| Generator | Discriminator|
|-----|-----------|--------------|
|DeBERTa-V3-XSmall|[Hugging Face](https://huggingface.co/microsoft/deberta-v3-xsmall/resolve/main/pytorch_model.generator.bin)     |[Hugging Face](https://huggingface.co/microsoft/deberta-v3-xsmall/resolve/main/pytorch_model.bin) |
|DeBERTa-V3-Small|[Hugging Face](https://huggingface.co/microsoft/deberta-v3-small/resolve/main/pytorch_model.generator.bin)     |[Hugging Face](https://huggingface.co/microsoft/deberta-v3-small/resolve/main/pytorch_model.bin) |
|DeBERTa-V3-Large|[Hugging Face](https://huggingface.co/microsoft/deberta-v3-large/resolve/main/pytorch_model.generator.bin)     |[Hugging Face](https://huggingface.co/microsoft/deberta-v3-large/resolve/main/pytorch_model.bin) |

## Distributed training

To train with multiple node, you need to specify three environment variables,

`WORLD_SIZE` - Total nodes that are used for the training

`MASTER_ADDR` - The IP address or host name of the master node

`MASTER_PORT` - The port of the master node

`RANK` - The rank of current node

For example, to run train a model with 2 nodes,

- On **node0**, 
 ``` bash
 export WORLD_SIZE=2
 export MASTER_ADDR=node0
 export MASTER_PORT=7488
 export RANK=0
 ./rtd.sh deberta-v3-xsmall
 ```

- On **node1**, 
 ``` bash
 export WORLD_SIZE=2
 export MASTER_ADDR=node0
 export MASTER_PORT=7488
 export RANK=1
 ./rtd.sh deberta-v3-xsmall
 ```

## Model config options

- `embedding_sharing` The embedding sharing method
	- **GDES** Gradient Disentangled Embedding Sharing
	- **ES**  Embedding Sharing
	- **NES** No Embedding Sharing

- `relative_attention` Whether to used relative attention

- `pos_att_type` Relative position encoding type
	- P2C Postion to content attention
	- C2P Content to position attention
