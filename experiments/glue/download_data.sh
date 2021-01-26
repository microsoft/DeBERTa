#!/bin/bash

cache_dir=$1
task=$2
if [[ -z $cache_dir ]]; then
	cache_dir=/tmp/DeBERTa/glue 
fi


mkdir -p $cache_dir
curl -s -J -L  https://raw.githubusercontent.com/nyu-mll/jiant/v1.3.2/scripts/download_glue_data.py -o $cache_dir/glue.py
patch $cache_dir/glue.py patch.diff
if [[ -z $task ]]; then
	python3 $cache_dir/glue.py  --data_dir $cache_dir/
else
	python3 $cache_dir/glue.py  --data_dir $cache_dir/ --tasks $task
fi
