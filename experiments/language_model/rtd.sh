#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=/tmp/DeBERTa/RTD/

max_seq_length=512
data_dir=$cache_dir/wiki103/spm_$max_seq_length

function setup_wiki_data(){
	task=$1
	mkdir -p $cache_dir
	if [[ ! -e  $cache_dir/spm.model ]]; then
		wget -q https://huggingface.co/microsoft/deberta-v3-base/resolve/main/spm.model -O $cache_dir/spm.model
	fi

	if [[ ! -e  $data_dir/test.txt ]]; then
		wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -O $cache_dir/wiki103.zip
		unzip -j $cache_dir/wiki103.zip -d $cache_dir/wiki103
		mkdir -p $data_dir
		python ./prepare_data.py -i $cache_dir/wiki103/wiki.train.tokens -o $data_dir/train.txt --max_seq_length $max_seq_length
		python ./prepare_data.py -i $cache_dir/wiki103/wiki.valid.tokens -o $data_dir/valid.txt --max_seq_length $max_seq_length
		python ./prepare_data.py -i $cache_dir/wiki103/wiki.test.tokens -o $data_dir/test.txt --max_seq_length $max_seq_length
	fi
}

setup_wiki_data

Task=RTD

init=$1
tag=$init
case ${init,,} in
	deberta-v3-xsmall-continue)
	# wget https://huggingface.co/microsoft/deberta-v3-xsmall/resolve/main/pytorch_model.generator.bin
	# wget https://huggingface.co/microsoft/deberta-v3-xsmall/resolve/main/pytorch_model.bin
	parameters=" --num_train_epochs 1 \
	--model_config rtd_xsmall.json \
	--warmup 10000 \
	--num_training_steps 100000 \
	--learning_rate 5e-5 \
	--train_batch_size 256 \
	--init_generator <TODO: generator checkpoint> \
	--init_discriminator <TODO: discriminator checkpoint> \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-xsmall)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_xsmall.json \
	--warmup 10000 \
	--learning_rate 3e-4 \
	--train_batch_size 64 \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-small-continue)
	# wget https://huggingface.co/microsoft/deberta-v3-small/resolve/main/pytorch_model.generator.bin
	# wget https://huggingface.co/microsoft/deberta-v3-small/resolve/main/pytorch_model.bin
	parameters=" --num_train_epochs 1 \
	--model_config rtd_small.json \
	--warmup 10000 \
	--num_training_steps 100000 \
	--learning_rate 5e-5 \
	--train_batch_size 256 \
	--init_generator <TODO: generator checkpoint> \
	--init_discriminator <TODO: discriminator checkpoint> \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-base)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_base.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-large)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_large.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--decoupled_training True \
	--fp16 True "
		;;
	*)
		echo "usage $0 <Pretrained model configuration>"
		echo "Supported configurations"
		echo "deberta-v3-xsmall - Pretrained DeBERTa v3 XSmall model with 9M backbone network parameters (12 layers, 256 hidden size) plus 32M embedding parameters(128k vocabulary size)"
		echo "deberta-v3-xsmall - Pretrained DeBERTa v3 Base model with 81M backbone network parameters (12 layers, 768 hidden size) plus 96M embedding parameters(128k vocabulary size)"
		echo "deberta-v3-xsmall - Pretrained DeBERTa v3 Large model with 288M backbone network parameters (24 layers, 1024 hidden size) plus 128M embedding parameters(128k vocabulary size)"
		exit 0
		;;
esac

python -m DeBERTa.apps.run --model_config config.json  \
	--tag $tag \
	--do_train \
	--num_training_steps 1000000 \
	--max_seq_len $max_seq_length \
	--dump 10000 \
	--task_name $Task \
	--data_dir $data_dir \
	--vocab_path $cache_dir/spm.model \
	--vocab_type spm \
	--output_dir /tmp/ttonly/$tag/$task  $parameters
