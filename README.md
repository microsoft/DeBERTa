# DeBERTa: Decoding-enhanced BERT with Disentangled Attention

This repository is the official implementation of [ **DeBERTa**: **D**ecoding-**e**nhanced **BERT** with Disentangled **A**ttention ](https://arxiv.org/abs/2006.03654)

## News
### 06/13/2020
We released the pre-trained models, source code, and fine-tuning scripts to reproduce some of the experimental results in the paper. You can follow similar scripts to apply DeBERTa to your own experiments or applications. Pre-training scripts will be released in the next step. 

## Introduction to DeBERTa 
DeBERTa (Decoding-enhanced BERT with disentangled attention) improves the BERT and RoBERTa models using two novel techniques. The first is the disentangled attention mechanism, where each word is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices on their contents and relative positions. Second, an enhanced mask decoder is used to replace the output softmax layer to predict the masked tokens for model pretraining. We show that these two techniques significantly improve the efficiency of model pre-training and performance of downstream tasks.

# Pre-trained Models

Our pre-trained models are packaged into zipped files. You can download them from our [releasements](https://github.com/microsoft/DeBERTa/releases), or download an individual model via the links below:
- [Large](https://github.com/microsoft/DeBERTa/releases/download/v0.1/large.zip): the pre-trained Large model
- [Base](https://github.com/microsoft/DeBERTa/releases/download/v0.1/base.zip) : the pre-trained Base model
- [Large MNLI](https://github.com/microsoft/DeBERTa/releases/download/v0.1/large_mnli.zip): Large model fine-tuned with MNLI task
- [Base MNLI](https://github.com/microsoft/DeBERTa/releases/download/v0.1/base_mnli.zip): Base model fine-tuned with MNLI task


# Try the code

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
    self.bert = deberta.DeBERTa(pre_trained='base') # Or 'large' or 'base_mnli' or 'large_mnli'
    # Your existing model code
    # do inilization as before
    # 
    self.bert.apply_state() # Apply the pre-trained model of DeBERTa at the end of the constructor
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
    encoding = self.bert(input_ids)[-1]

# 2. Change your tokenizer with the the tokenizer built in DeBERta
from DeBERTa import deberta
tokenizer = deberta.GPT2Tokenizer()
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
curl -J -L https://raw.githubusercontent.com/nyu-mll/jiant/master/scripts/download_glue_data.py | python3 - --data_dir $cache_dir/glue_tasks
```

2. Run task

``` bash
task=STS-B 
OUTPUT=/tmp/DeBERTa/exps/$task
export OMP_NUM_THREADS=1
python3 -m DeBERTa.apps.train --task_name $task --do_train  \
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

## Important Notes
1. To run our code on multiple GPUs, you must `OMP_NUM_THREADS=1` before lunch our training code
2. By default we will cache the pre-trained model and tokenizer at `$HOME/.~DeBERTa`, you may need to clean it if the downloading failed unexpectedly.


## Experiments
Our fine-tuning experiments are carried on half a DGX-2 node with 8x32 V100 GPU cards, the results may vary due to different GPU models, drivers, CUDA SDK versions, using FP16 or FP32, and random seeds. 
We report our numbers based on multple runs with different random seeds here. Here are the results from the Large model:

|Task	 |Command	|Results	|Running Time(8x32G V100 GPUs)|
|--------|---------------|---------------|-------------------------|
|MNLI xlarge|	`experiments/glue/mnli_xlarge.sh`|	91.5/91.4 +/-0.1|	2.5h|
|MNLI large|	`experiments/glue/mnli_large.sh`|	91.2/91.0 +/-0.1|	2.5h|
|QQP large|	`experiments/glue/qqp_large.sh`|	92.3 +/-0.1|		6h|
|QNLI large|	`experiments/glue/qnli_large.sh`|	95.3 +/-0.2|		2h|
|MRPC large|	`experiments/glue/mrpc_large.sh`|	93.4 +/-0.5|		0.5h|
|RTE large|	`experiments/glue/rte_large.sh`|	87.7 +/-1.0|		0.5h|
|SST-2 large|	`experiments/glue/sst2_large.sh`|	96.7 +/-0.3|		1h|
|STS-b large|	`experiments/glue/Stsb_large.sh`|	92.5 +/-0.3|		0.5h|
|CoLA large|	`experiments/glue/cola_large.sh`|	70.5 +/-1.0|		0.5h|

And here are the results from the Base model

|Task	 |Command	|Results	|Running Time(8x32G V100 GPUs)|
|--------|---------------|---------------|-------------------------|
|MNLI base|	`experiments/glue/mnli_base.sh`|	88.8/88.5 +/-0.2|	1.5h|

## Contacts

Pengcheng He(penhe@microsoft.com), Xiaodong Liu(xiaodl@microsoft.com), Jianfeng Gao(jfgao@microsoft.com), Weizhu Chen(wzchen@microsoft.com)

# Citation
```
@misc{he2020deberta,
    title={DeBERTa: Decoding-enhanced BERT with Disentangled Attention},
    author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
    year={2020},
    eprint={2006.03654},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

