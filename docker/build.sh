#!/bin/bash
work_dir=/tmp/DeBERTa/~docker~/
rm -rf $work_dir
mkdir -p $work_dir
cp Dockerfile $work_dir
cp ../requirements.txt $work_dir
docker build -f $work_dir/Dockerfile $work_dir -t deberta
