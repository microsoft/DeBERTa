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

init=large 

tag=Large
Task=CoLA

setup_glue_data $Task

python -m DeBERTa.apps.run --model_config config.json  \
	--tag $tag \
	--task_name $Task \
	--data_dir $cache_dir/glue_tasks/$Task \
	--init_model $init \
	--output_dir /tmp/ttonly/$tag/$task \
	--num_train_epochs 5 \
	--warmup 100 \
	--learning_rate 1e-5 \
	--train_batch_size 32 \
	--cls_drop_out 0.15 \
	--do_train \
	--max_seq_len 64 
