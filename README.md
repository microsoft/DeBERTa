# DeBERTa: Decoding-enhanced BERT with Disentangled Attention

This repository is the official implementation of [ **DeBERTa**: **D**ecoding-**e**nhanced **BERT** with Disentangled **A**ttention ](https://arxiv.org/abs/2006.03654) and [DeBERTa V3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543)

## News
### 12/8/2021
- [DeBERTa-V3-XSmall](https://huggingface.co/microsoft/deberta-v3-xsmall) is added. With only **22M** backbone parameters which is only 1/4 of RoBERTa-Base and XLNet-Base, DeBERTa-V3-XSmall significantly outperforms the later on MNLI and SQuAD v2.0 tasks (i.e. 1.2% on MNLI-m, 1.5% EM score on SQuAD v2.0). This further demnostrates the efficiency of DeBERTaV3 models.

### 11/16/2021
- The models of our new work [DeBERTa V3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543) are publicly available at [huggingface model hub](https://huggingface.co/models?other=deberta-v3) now. The new models are based on DeBERTa-V2 models by replacing MLM with ELECTRA-style objective plus gradient-disentangled embedding sharing which further improves the model efficiency.
- Scripts for DeBERTa V3 model fine-tuning are added
- Code of RTD task head is added
- [Document](experiments/language_model) for pre-training is added

### 3/31/2021
- Masked language model task is added
- SuperGLUE tasks is added
- SiFT code is added

### 2/03/2021
DeBERTa v2 code and the **900M, 1.5B** [model](https://huggingface.co/models?search=microsoft%2Fdeberta) are here now. This includes the 1.5B model used for our SuperGLUE single-model submission and achieving 89.9, versus human baseline 89.8. You can find more details about this submission in our [blog](https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/)

#### What's new in v2
- **Vocabulary** In v2 we use a new vocabulary of size 128K built from the training data. Instead of GPT2 tokenizer, we use [sentencepiece](https://github.com/google/sentencepiece) tokenizer.
- **nGiE(nGram Induced Input Encoding)** In v2 we use an additional convolution layer aside with the first transformer layer to better learn the local dependency of input tokens. We will add more ablation studies on this feature.
- **Sharing position projection matrix with content projection matrix in attention layer** Based on our previous experiment, we found this can save parameters without affecting the performance.
- **Apply bucket to encode relative postions** In v2 we use log bucket to encode relative positions similar to T5. 
- **900M model & 1.5B model** In v2 we scale our model size to 900M and 1.5B which significantly improves the performance of downstream tasks.

### 12/29/2020
With DeBERTa 1.5B model, we surpass T5 11B model and human performance on SuperGLUE leaderboard. Code and model will be released soon. Please check out our paper for more details.

### 06/13/2020
We released the pre-trained models, source code, and fine-tuning scripts to reproduce some of the experimental results in the paper. You can follow similar scripts to apply DeBERTa to your own experiments or applications. Pre-training scripts will be released in the next step. 


## Introduction to DeBERTa 
DeBERTa (Decoding-enhanced BERT with disentangled attention) improves the BERT and RoBERTa models using two novel techniques. The first is the disentangled attention mechanism, where each word is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices on their contents and relative positions. Second, an enhanced mask decoder is used to replace the output softmax layer to predict the masked tokens for model pretraining. We show that these two techniques significantly improve the efficiency of model pre-training and performance of downstream tasks.

# Pre-trained Models

Our pre-trained models are packaged into zipped files. You can download them from our [releases](https://huggingface.co/models?search=microsoft%2Fdeberta), or download an individual model via the links below:

|Model        | Vocabulary(K)|Backbone Parameters(M)| Hidden Size | Layers| Note|
|-------------|------|---------|-----|-----|---------|
|**[V2-XXLarge](https://huggingface.co/microsoft/deberta-v2-xxlarge)<sup>1</sup>**|128|1320|1536| 48|128K new SPM vocab |
|[V2-XLarge](https://huggingface.co/microsoft/deberta-v2-xlarge)|128|710|1536| 24| 128K new SPM vocab|
|[XLarge](https://huggingface.co/microsoft/deberta-xlarge)|50|700|1024|48| Same vocab as RoBERTa|
|[Large](https://huggingface.co/microsoft/deberta-large)|50|350|1024|24|Same vocab as RoBERTa|
|[Base](https://huggingface.co/microsoft/deberta-base)|50|100|768|12|Same vocab as RoBERTa|
|[V2-XXLarge-MNLI](https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli)|128|1320|1536| 48|Fine-turned with MNLI |
|[V2-XLarge-MNLI](https://huggingface.co/microsoft/deberta-v2-xlarge-mnli)|128|710|1536| 24|Fine-turned with MNLI |
|[XLarge-MNLI](https://huggingface.co/microsoft/deberta-xlarge-mnli)|50|700|1024|48|Fine-turned with MNLI|
|[Large-MNLI](https://huggingface.co/microsoft/deberta-large-mnli)|50|350|1024|24|Fine-turned with MNLI|
|[Base-MNLI](https://huggingface.co/microsoft/deberta-base-mnli)|50|86|768|12|Fine-turned with MNLI|
|[DeBERTa-V3-Large](https://huggingface.co/microsoft/deberta-v3-large)<sup>2</sup>|128|304|1024| 24| 128K new SPM vocab|
|[DeBERTa-V3-Base](https://huggingface.co/microsoft/deberta-v3-base)<sup>2</sup>|128|86|768| 12| 128K new SPM vocab|
|[DeBERTa-V3-Small](https://huggingface.co/microsoft/deberta-v3-small)<sup>2</sup>|128|44|768| 6| 128K new SPM vocab|
|[DeBERTa-V3-XSmall](https://huggingface.co/microsoft/deberta-v3-xsmall)<sup>2</sup>|128|22|384| 12| 128K new SPM vocab|
|[mDeBERTa-V3-Base](https://huggingface.co/microsoft/mdeberta-v3-base)<sup>2</sup>|250|86|768| 12| 250K new SPM vocab, multi-lingual model with 102 languages|

## Note 
- 1 This is the model(89.9) that surpassed **T5 11B(89.3) and human performance(89.8)** on **SuperGLUE** for the first time. 128K new SPM vocab. 
- 2 These V3 DeBERTa models are deberta models pre-trained with ELECTRA-style objective plus gradient-disentangled embedding sharing which significantly improves the model efficiency.


# Try the model

Read our [documentation](https://deberta.readthedocs.io/en/latest/)

## Requirements
- Linux system, e.g. Ubuntu 18.04LTS
- CUDA 10.0
- pytorch 1.3.0
- python 3.6
- bash shell 4.0
- curl
- docker (optional)
- nvidia-docker2 (optional)

There are several ways to try our code,
### Use docker

Docker is the recommended way to run the code as we already built every dependency into the our docker [bagai/deberta](https://hub.docker.com/r/bagai/deberta) and you can follow the [docker official site](https://docs.docker.com/engine/install/ubuntu/) to install docker on your machine.

To run with docker, make sure your system fullfil the requirements in the above list. Here are the steps to try the GLUE experiments: Pull the code, run `./run_docker.sh` 
, and then you can run the bash commands under `/DeBERTa/experiments/glue/`


### Use pip
Pull the code and run `pip3 install -r requirements.txt` in the root directory of the code, then enter `experiments/glue/` folder of the code and try the bash commands under that folder for glue experiments.

### Install as a pip package
`pip install deberta`

#### Use DeBERTa in existing code
``` Python

# To apply DeBERTa into your existing code, you need to make two changes on your code,
# 1. change your model to consume DeBERTa as the encoder
from DeBERTa import deberta
import torch
class MyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    # Your existing model code
    self.deberta = deberta.DeBERTa(pre_trained='base') # Or 'large' 'base-mnli' 'large-mnli' 'xlarge' 'xlarge-mnli' 'xlarge-v2' 'xxlarge-v2'
    # Your existing model code
    # do inilization as before
    # 
    self.deberta.apply_state() # Apply the pre-trained model of DeBERTa at the end of the constructor
    #
  def forward(self, input_ids):
    # The inputs to DeBERTa forward are
    # `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary
    # `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices selected in [0, 1]. 
    #    Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
    # `attention_mask`: an optional parameter for input mask or attention mask. 
    #   - If it's an input mask, then it will be torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1]. 
    #      It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch. 
    #      It's the mask that we typically use for attention when a batch has varying length sentences.
    #   - If it's an attention mask then if will be torch.LongTensor of shape [batch_size, sequence_length, sequence_length]. 
    #      In this case, it's a mask indicate which tokens in the sequence should be attended by other tokens in the sequence. 
    # `output_all_encoded_layers`: whether to output results of all encoder layers, default, True
    encoding = deberta.bert(input_ids)[-1]

# 2. Change your tokenizer with the the tokenizer built in DeBERta
from DeBERTa import deberta
vocab_path, vocab_type = deberta.load_vocab(pretrained_id='base')
tokenizer = deberta.tokenizers[vocab_type](vocab_path)
# We apply the same schema of special tokens as BERT, e.g. [CLS], [SEP], [MASK]
max_seq_len = 512
tokens = tokenizer.tokenize('Examples input text of DeBERTa')
# Truncate long sequence
tokens = tokens[:max_seq_len -2]
# Add special tokens to the `tokens`
tokens = ['[CLS]'] + tokens + ['[SEP]']
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_mask = [1]*len(input_ids)
# padding
paddings = max_seq_len-len(input_ids)
input_ids = input_ids + [0]*paddings
input_mask = input_mask + [0]*paddings
features = {
'input_ids': torch.tensor(input_ids, dtype=torch.int),
'input_mask': torch.tensor(input_mask, dtype=torch.int)
}

```

#### Run DeBERTa experiments from command line
For glue tasks, 
1. Get the data
``` bash
cache_dir=/tmp/DeBERTa/
cd experiments/glue
./download_data.sh  $cache_dir/glue_tasks
```

2. Run task

``` bash
task=STS-B 
OUTPUT=/tmp/DeBERTa/exps/$task
export OMP_NUM_THREADS=1
python3 -m DeBERTa.apps.run --task_name $task --do_train  \
  --data_dir $cache_dir/glue_tasks/$task \
  --eval_batch_size 128 \
  --predict_batch_size 128 \
  --output_dir $OUTPUT \
  --scale_steps 250 \
  --loss_scale 16384 \
  --accumulative_update 1 \  
  --num_train_epochs 6 \
  --warmup 100 \
  --learning_rate 2e-5 \
  --train_batch_size 32 \
  --max_seq_len 128
```

## Notes
- 1. By default we will cache the pre-trained model and tokenizer at `$HOME/.~DeBERTa`, you may need to clean it if the downloading failed unexpectedly.
- 2. You can also try our models with [HF Transformers](https://github.com/huggingface/transformers). But when you try XXLarge model you need to specify --sharded_ddp argument. Please check our [XXLarge model card](https://huggingface.co/microsoft/deberta-xxlarge-v2) for more details.

## Experiments
Our fine-tuning experiments are carried on half a DGX-2 node with 8x32 V100 GPU cards, the results may vary due to different GPU models, drivers, CUDA SDK versions, using FP16 or FP32, and random seeds. 
We report our numbers based on multple runs with different random seeds here. Here are the results from the Large model:

|Task	 |Command	|Results	|Running Time(8x32G V100 GPUs)|
|--------|---------------|---------------|-------------------------|
|**MNLI xxlarge v2**|	`experiments/glue/mnli.sh xxlarge-v2`|	**91.7/91.9** +/-0.1|	4h|
|MNLI xlarge v2|	`experiments/glue/mnli.sh xlarge-v2`|	91.7/91.6 +/-0.1|	2.5h|
|MNLI xlarge|	`experiments/glue/mnli.sh xlarge`|	91.5/91.2 +/-0.1|	2.5h|
|MNLI large|	`experiments/glue/mnli.sh large`|	91.3/91.1 +/-0.1|	2.5h|
|QQP large|	`experiments/glue/qqp.sh large`|	92.3 +/-0.1|		6h|
|QNLI large|	`experiments/glue/qnli.sh large`|	95.3 +/-0.2|		2h|
|MRPC large|	`experiments/glue/mrpc.sh large`|	91.9 +/-0.5|		0.5h|
|RTE large|	`experiments/glue/rte.sh large`|	86.6 +/-1.0|		0.5h|
|SST-2 large|	`experiments/glue/sst2.sh large`|	96.7 +/-0.3|		1h|
|STS-b large|	`experiments/glue/Stsb.sh large`|	92.5 +/-0.3|		0.5h|
|CoLA large|	`experiments/glue/cola.sh`|	70.5 +/-1.0|		0.5h|

And here are the results from the Base model

|Task	 |Command	|Results	|Running Time(8x32G V100 GPUs)|
|--------|---------------|---------------|-------------------------|
|MNLI base|	`experiments/glue/mnli.sh base`|	88.8/88.5 +/-0.2|	1.5h|


### Fine-tuning on NLU tasks

We present the dev results on SQuAD 1.1/2.0 and several GLUE benchmark tasks.

| Model                     | SQuAD 1.1 | SQuAD 2.0 | MNLI-m/mm   | SST-2 | QNLI | CoLA | RTE    | MRPC  | QQP   |STS-B |
|---------------------------|-----------|-----------|-------------|-------|------|------|--------|-------|-------|------|
|                           | F1/EM     | F1/EM     | Acc         | Acc   | Acc  | MCC  | Acc    |Acc/F1 |Acc/F1 |P/S   |
| BERT-Large                | 90.9/84.1 | 81.8/79.0 | 86.6/-      | 93.2  | 92.3 | 60.6 | 70.4   | 88.0/-       | 91.3/- |90.0/- |
| RoBERTa-Large             | 94.6/88.9 | 89.4/86.5 | 90.2/-      | 96.4  | 93.9 | 68.0 | 86.6   | 90.9/-       | 92.2/- |92.4/- |
| XLNet-Large               | 95.1/89.7 | 90.6/87.9 | 90.8/-      | 97.0  | 94.9 | 69.0 | 85.9   | 90.8/-       | 92.3/- |92.5/- |
| [DeBERTa-Large](https://huggingface.co/microsoft/deberta-large)<sup>1</sup> | 95.5/90.1 | 90.7/88.0 | 91.3/91.1| 96.5|95.3| 69.5| 91.0| 92.6/94.6| 92.3/- |92.8/92.5 |
| [DeBERTa-XLarge](https://huggingface.co/microsoft/deberta-xlarge)<sup>1</sup> | -/-  | -/-  | 91.5/91.2| 97.0 | - | -    | 93.1   | 92.1/94.3    | -    |92.9/92.7|
| [DeBERTa-V2-XLarge](https://huggingface.co/microsoft/deberta-v2-xlarge)<sup>1</sup>|95.8/90.8| 91.4/88.9|91.7/91.6| **97.5**| 95.8|71.1|**93.9**|92.0/94.2|92.3/89.8|92.9/92.9|
|**[DeBERTa-V2-XXLarge](https://huggingface.co/microsoft/deberta-v2-xxlarge)<sup>1,2</sup>**|**96.1/91.4**|**92.2/89.7**|**91.7/91.9**|97.2|**96.0**|72.0| 93.5| **93.1/94.9**|**92.7/90.3** |**93.2/93.1** |
|**[DeBERTa-V3-Large](https://huggingface.co/microsoft/deberta-v3-large)**|-/-|91.5/89.0|**91.8/91.9**|96.9|**96.0**|**75.3**| 92.7| 92.2/-|**93.0/-** |93.0/- |
|[DeBERTa-V3-Base](https://huggingface.co/microsoft/deberta-v3-base)|-/-|88.4/85.4|90.6/90.7|-|-|-| -| -|- |- |
|[DeBERTa-V3-Small](https://huggingface.co/microsoft/deberta-v3-small)|-/-|82.9/80.4|88.3/87.7|-|-|-| -| -|- |- |
|[DeBERTa-V3-XSmall](https://huggingface.co/microsoft/deberta-v3-xsmall)|-/-|84.8/82.0|88.1/88.3|-|-|-| -| -|- |- |

### Fine-tuning on XNLI

We present the dev results on XNLI with zero-shot crosslingual transfer setting, i.e. training with english data only, test on other languages.

| Model        |avg | en |  fr| es  | de  | el  | bg  | ru  |tr   |ar   |vi   | th  | zh | hi  | sw  | ur  | 
|--------------| ----|----|----|---- |--   |--   |--   | --  |--   |--   |--   | --  | -- | --  | --  | --  |
| XLM-R-base   |76.2 |85.8|79.7|80.7 |78.7 |77.5 |79.6 |78.1 |74.2 |73.8 |76.5 |74.6 |76.7| 72.4| 66.5| 68.3|
| [mDeBERTa-V3-Base](https://huggingface.co/microsoft/mdeberta-v3-base)|**79.8**+/-0.2|**88.2**|**82.6**|**84.4** |**82.7** |**82.3** |**82.4** |**80.8** |**79.5** |**78.5** |**78.1** |**76.4** |**79.5**| **75.9**| **73.9**| **72.4**|

--------
#### Notes.
 - <sup>1</sup> Following RoBERTa, for RTE, MRPC, STS-B, we fine-tune the tasks based on [DeBERTa-Large-MNLI](https://huggingface.co/microsoft/deberta-large-mnli), [DeBERTa-XLarge-MNLI](https://huggingface.co/microsoft/deberta-xlarge-mnli), [DeBERTa-V2-XLarge-MNLI](https://huggingface.co/microsoft/deberta-v2-xlarge-mnli), [DeBERTa-V2-XXLarge-MNLI](https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli). The results of SST-2/QQP/QNLI/SQuADv2 will also be slightly improved when start from MNLI fine-tuned models, however, we only report the numbers fine-tuned from pretrained base models for those 4 tasks.

### Pre-training with MLM and RTD objectives

To pre-train DeBERTa with MLM and RTD objectives, please check [`experiments/language_models`](experiments/language_model)

 
## Contacts

Pengcheng He(penhe@microsoft.com), Xiaodong Liu(xiaodl@microsoft.com), Jianfeng Gao(jfgao@microsoft.com), Weizhu Chen(wzchen@microsoft.com)

# Citation
``` latex
@misc{he2021debertav3,
      title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing}, 
      author={Pengcheng He and Jianfeng Gao and Weizhu Chen},
      year={2021},
      eprint={2111.09543},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

``` latex
@inproceedings{
he2021deberta,
title={DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION},
author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=XPZIaotutsD}
}
```

