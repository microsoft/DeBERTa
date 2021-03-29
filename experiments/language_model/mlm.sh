#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=/tmp/DeBERTa/MLM/

function setup_wiki_data(){
	task=$1
	mkdir -p $cache_dir
	if [[ ! -e  $cache_dir/spm.model ]]; then
		wget -q https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/spm.model -O $cache_dir/spm.model
	fi

	if [[ ! -e  $cache_dir/wiki103.zip ]]; then
		wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -O $cache_dir/wiki103.zip
		unzip -j $cache_dir/wiki103.zip -d $cache_dir/wiki103
		mkdir -p $cache_dir/wiki103/spm
		python ./prepare_data.py -i $cache_dir/wiki103/wiki.train.tokens -o $cache_dir/wiki103/spm/train.txt
		python ./prepare_data.py -i $cache_dir/wiki103/wiki.valid.tokens -o $cache_dir/wiki103/spm/valid.txt
		python ./prepare_data.py -i $cache_dir/wiki103/wiki.test.tokens -o $cache_dir/wiki103/spm/test.txt
	fi
}

setup_wiki_data

Task=MLM

init=$1
tag=$init
case ${init,,} in
	xlarge-v2)
	parameters=" --num_train_epochs 1 \
	--model_config xlarge.json \
	--warmup 1000 \
	--learning_rate 1e-4 \
	--train_batch_size 32 \
	--max_ngram 3 \
	--fp16 True "
		;;
	xxlarge-v2)
	parameters=" --num_train_epochs 1 \
	--warmup 1000 \
	--model_config xxlarge.json \
	--learning_rate 1e-4 \
	--train_batch_size 32 \
	--max_ngram 3 \
	--fp16 True "
		;;
	*)
		echo "usage $0 <Pretrained model configuration>"
		echo "Supported configurations"
		echo "xlarge-v2 - Pretrained DeBERTa v2 model with 900M parameters (24 layers, 1536 hidden size)"
		echo "xxlarge-v2 - Pretrained DeBERTa v2 model with 1.5B parameters (48 layers, 1536 hidden size)"
		exit 0
		;;
esac

data_dir=$cache_dir/wiki103/spm
python -m DeBERTa.apps.run --model_config config.json  \
	--tag $tag \
	--do_train \
	--num_training_steps 10000 \
	--max_seq_len 512 \
	--task_name $Task \
	--data_dir $data_dir \
	--vocab_path $cache_dir/spm.model \
	--vocab_type spm \
	--output_dir /tmp/ttonly/$tag/$task  $parameters
