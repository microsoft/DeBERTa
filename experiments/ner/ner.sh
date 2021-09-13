#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=/tmp/DeBERTa/

data_dir=$cache_dir/NER/data
function get_data(){
	mkdir -p $data_dir
	if [[ ! -e $data_dir/train.txt ]]; then
		pip install seqeval
		curl -L https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train -o "$data_dir/train.txt"
		curl -L https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa -o "$data_dir/valid.txt"
		curl -L https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb -o "$data_dir/test.txt"
	fi
}

Task=NER
get_data

init=$1
tag=$init
case ${init,,} in
	base)
	parameters=" --num_train_epochs 15 \
	--warmup 0.1 \
	--learning_rate 2e-5 \
	--train_batch_size 16 \
	--cls_drop_out 0 "
		;;
	large)
	parameters=" --num_train_epochs 15 \
	--warmup 0.1 \
	--learning_rate 1e-5 \
	--train_batch_size 16 \
	--cls_drop_out 0 \
	--fp16 True "
		;;
	xlarge)
	parameters=" --num_train_epochs 15 \
	--warmup 0.1 \
	--learning_rate 7e-6 \
	--train_batch_size 16 \
	--cls_drop_out 0 \
	--fp16 True "
		;;
	xlarge-v2)
	parameters=" --num_train_epochs 15 \
	--warmup 0.1 \
	--learning_rate 4e-6 \
	--train_batch_size 16 \
	--cls_drop_out 0 \
	--fp16 True "
		;;
	xxlarge-v2)
	parameters=" --num_train_epochs 15 \
	--warmup 0.1 \
	--learning_rate 2.5e-6 \
	--train_batch_size 16 \
	--cls_drop_out 0 \
	--fp16 True "
		;;
	deberta-v3-large)
	parameters=" --num_train_epochs 10 \
	--warmup 10 \
	--learning_rate 9e-6 \
	--fp16 True \
	--train_batch_size 16 \
	--cls_drop_out 0 "
		;;
	*)
		echo "usage $0 <Pretrained model configuration>"
		echo "Supported configurations"
		echo "base - Pretrained DeBERTa v1 model with 140M parameters (12 layers, 768 hidden size)"
		echo "large - Pretrained DeBERta v1 model with 380M parameters (24 layers, 1024 hidden size)"
		echo "xlarge - Pretrained DeBERTa v1 model with 750M parameters (48 layers, 1024 hidden size)"
		echo "xlarge-v2 - Pretrained DeBERTa v2 model with 900M parameters (24 layers, 1536 hidden size)"
		echo "xxlarge-v2 - Pretrained DeBERTa v2 model with 1.5B parameters (48 layers, 1536 hidden size)"
		exit 0
		;;
esac

python -m DeBERTa.apps.run --model_config config.json  \
	--tag $tag \
	--do_train \
	--max_seq_len 512 \
	--task_name $Task \
	--data_dir $data_dir \
	--init_model $init \
	--output_dir /tmp/ttonly/$tag/$task  $parameters

