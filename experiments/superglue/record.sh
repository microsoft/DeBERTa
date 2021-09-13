#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=/tmp/DeBERTa/

function setup_data(){
	task=$1
	mkdir -p $cache_dir
	if [[ ! -e $cache_dir/superglue_tasks/${task}/train.jsonl ]]; then
		./download_data.sh $cache_dir/superglue_tasks
	fi
}


Task=ReCoRD
setup_data $Task

init=$1
tag=$init
case ${init,,} in
	base)
	parameters=" --num_train_epochs 3 \
	--warmup 1000 \
	--learning_rate 2e-5 \
	--train_batch_size 64 \
	--cls_drop_out 0.3 "
		;;
	large)
	parameters=" --num_train_epochs 3 \
	--warmup 1000 \
	--learning_rate 1e-5 \
	--train_batch_size 64 \
	--cls_drop_out 0.3 \
	--fp16 True "
		;;
	xlarge)
	parameters=" --num_train_epochs 3 \
	--warmup 1000 \
	--learning_rate 7e-6 \
	--train_batch_size 64 \
	--cls_drop_out 0.3 \
	--fp16 True "
		;;
	xlarge-v2)
	parameters=" --num_train_epochs 3 \
	--warmup 1000 \
	--learning_rate 5e-6 \
	--train_batch_size 64 \
	--cls_drop_out 0.15 \
	--fp16 True "
		;;
	xxlarge-v2)
	parameters=" --num_train_epochs 3 \
	--warmup 1000 \
	--learning_rate 3e-6 \
	--accumulative_update 2 \
	--train_batch_size 64 \
	--cls_drop_out 0.15 \
	--fp16 True "
		;;
	deberta-v3-large)
	parameters=" --num_train_epochs 3 \
	--fp16 True \
	--warmup 500 \
	--learning_rate 7e-6 \
	--train_batch_size 64 \
	--cls_drop_out 0.1 "
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
	--data_dir $cache_dir/superglue_tasks/$Task \
	--init_model $init \
	--output_dir /tmp/ttonly/$tag/$task  $parameters
