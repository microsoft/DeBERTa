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

model_name=large
Task=NER
get_data
init=large
tag=large
../utils/train.sh --config config.json --vocab $cache_dir/vocab -t $Task --data $data_dir --tag $tag -i $init -o /tmp/ttonly/$tag/$Task -- --num_train_epochs 15 --accumulative_update 1 --warmup 0.1 --learning_rate 1e-5 --train_batch_size 16 --cls_drop 0  --max_seq_length 512
