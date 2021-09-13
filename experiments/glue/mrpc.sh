#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=/tmp/DeBERTa/

function setup_glue_data(){
	task=$1
	mkdir -p $cache_dir
	if [[ ! -e $cache_dir/glue_tasks/${task}/train.tsv ]]; then
		./download_data.sh $cache_dir/glue_tasks
	fi
}

Task=MRPC
setup_glue_data $Task

# The performance will be better when it's initialized with MNLI fine-tuned models
init=$1
tag=$init
case ${init,,} in
	base)
	init=${init}-mnli
	parameters=" --num_train_epochs 6 \
	--warmup 50 \
	--learning_rate 2e-5 \
	--train_batch_size 32 \
	--cls_drop_out 0.15 \
	--max_seq_len 128"
		;;
	large)
	init=${init}-mnli
	parameters=" --num_train_epochs 6 \
	--warmup 50 \
	--learning_rate 1e-5 \
	--train_batch_size 32 \
	--cls_drop_out 0.15 \
	--fp16 True \
	--max_seq_len 128 "
		;;
	xlarge)
	init=${init}-mnli
	parameters=" --num_train_epochs 8 \
	--warmup 50 \
	--learning_rate 7e-6 \
	--train_batch_size 32 \
	--cls_drop_out 0.3 \
	--fp16 True \
	--max_seq_len 128 "
		;;
	xlarge-v2)
	init=${init}-mnli
	parameters=" --num_train_epochs 6 \
	--warmup 50 \
	--learning_rate 3.5e-6 \
	--train_batch_size 32 \
	--cls_drop_out 0.2 \
	--fp16 True \
	--max_seq_len 128 "
		;;
	xxlarge-v2)
	init=${init}-mnli
	parameters=" --num_train_epochs 6 \
	--warmup 50 \
	--learning_rate 2e-6 \
	--train_batch_size 32 \
	--cls_drop_out 0.2 \
	--fp16 True \
	--max_seq_len 128 "
		;;
	deberta-v3-large)
	parameters=" --num_train_epochs 6 \
	--fp16 True \
	--warmup 50 \
	--learning_rate 6e-6  \
	--train_batch_size 32 \
	--max_seq_len 128     \
	--cls_drop_out 0.2 "
		;;
	*)
		echo "usage $0 <Pretrained model configuration>"
		echo "Supported configurations"
		echo "base - Pretrained DeBERTa v1 model with 140M parameters (12 layers, 768 hidden size)"
		echo "large - Pretrained DeBERta v1 model with 380M parameters (24 layers, 1024 hidden size)"
		echo "xlarge - Pretrained DeBERTa v1 model with 750M parameters (48 layers, 1024 hidden size)"
		echo "xlarge-v2-mnli - Pretrained DeBERTa v2 model with 900M parameters (24 layers, 1536 hidden size)"
		echo "xxlarge-v2-mnli - Pretrained DeBERTa v2 model with 1.5B parameters (48 layers, 1536 hidden size)"
		exit 0
		;;
esac

python -m DeBERTa.apps.run --model_config config.json  \
	--do_train \
	--tag $tag \
	--task_name $Task \
	--data_dir $cache_dir/glue_tasks/$Task \
	--init_model $init \
	--output_dir /tmp/ttonly/$tag/$task  $parameters
