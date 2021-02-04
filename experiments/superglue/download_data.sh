#!/bin/bash

cache_dir=$1
if [[ -z $cache_dir ]]; then
	cache_dir=/tmp/DeBERTa/superglue 
fi


mkdir -p $cache_dir
curl -s -J -L  https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip -o $cache_dir/super.zip
unzip $cache_dir/super.zip -d $cache_dir
