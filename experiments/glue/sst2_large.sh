#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=/tmp/DeBERTa/

function setup_glue_data(){
	task=$1
	mkdir -p $cache_dir
	if [[ ! -e $cache_dir/glue_tasks/${task}/train.tsv ]]; then
		curl -J -L https://raw.githubusercontent.com/nyu-mll/jiant/master/scripts/download_glue_data.py | python3 - --data_dir $cache_dir/glue_tasks
	fi
}

init=large 

tag=Large
Task=SST-2

setup_glue_data $Task
../utils/train.sh --config config.json  -t $Task --data $cache_dir/glue_tasks/$Task --tag $tag -i $init -o /tmp/ttonly/$tag/$task -- --num_train_epochs 6 --accumulative_update 1 --warmup 100 --learning_rate 1e-5 --train_batch_size 32 --max_seq_len 128
