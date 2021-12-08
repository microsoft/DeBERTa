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


Task=MNLI
setup_glue_data $Task

#export CUDA_VISIBLE_DEVICES=0

init=$1
tag=$init
case ${init,,} in
	bert-xsmall)
	init=/tmp/ttonly/bert-xsmall/discriminator/pytorch.model-1000000.bin
	vocab_type=spm
	vocab_path=/tmp/ttonly/bert-xsmall/discriminator/spm.model
	parameters=" --num_train_epochs 3 \
	--fp16 True \
	--warmup 1500 \
	--learning_rate 1e-4 \
	--vocab_type $vocab_type \
	--vocab_path $vocab_path \
	--train_batch_size 64 \
	--cls_drop_out 0.1 "
		;;
	deberta-v3-xsmall)
	parameters=" --num_train_epochs 3 \
	--fp16 True \
	--warmup 1000 \
	--learning_rate 4.5e-5 \
	--workers 2 \
	--train_batch_size 64 \
	--cls_drop_out 0.10 "
		;;
	deberta-v3-xsmall-sift)
	init=deberta-v3-xsmall
	parameters=" --num_train_epochs 10 \
	--fp16 True \
	--warmup 1500 \
	--learning_rate 5e-5 \
	--vat_lambda 3 \
	--vat_learning_rate 1e-4 \
	--vat_init_perturbation 1e-2 \
	--train_batch_size 64 \
	--cls_drop_out 0.10 "
		;;
	deberta-v3-small)
	parameters=" --num_train_epochs 3 \
	--fp16 True \
	--warmup 1500 \
	--learning_rate 4.5e-5 \
	--train_batch_size 64 \
	--cls_drop_out 0.20 "
		;;
	deberta-v3-small-sift)
	init=deberta-v3-small
	parameters=" --num_train_epochs 6 \
	--vat_lambda 5 \
	--vat_learning_rate 1e-4 \
	--vat_init_perturbation 1e-2 \
	--fp16 False \
	--warmup 1000 \
	--learning_rate 3.5e-5 \
	--train_batch_size 64 \
	--cls_drop_out 0.1 "
		;;
	deberta-v3-base)
	parameters=" --num_train_epochs 3 \
	--fp16 True \
	--warmup 1000 \
	--learning_rate 1.5e-5 \
	--train_batch_size 64 \
	--cls_drop_out 0.1 "
		;;
	deberta-v3-base-sift)
	init=deberta-v3-base
	parameters=" --num_train_epochs 6 \
	--vat_lambda 5 \
	--fp16 True \
	--vat_learning_rate 1e-4 \
	--vat_init_perturbation 1e-2 \
	--warmup 1000 \
	--learning_rate 1.5e-5 \
	--train_batch_size 64 \
	--cls_drop_out 0.1 "
		;;
	deberta-v3-large)
	parameters=" --num_train_epochs 2 \
	--fp16 True \
	--warmup 500 \
	--learning_rate 7e-6 \
	--train_batch_size 64 \
	--cls_drop_out 0.3 "
		;;
	base)
	parameters=" --num_train_epochs 3 \
	--fp16 True \
	--warmup 1000 \
	--learning_rate 3e-5 \
	--train_batch_size 64 \
	--cls_drop_out 0.1 "
		;;
	base-sift)
  init=base
	parameters=" --num_train_epochs 6 \
	--vat_lambda 5 \
	--vat_learning_rate 1e-4 \
	--vat_init_perturbation 1e-2 \
	--fp16 True \
	--warmup 1000 \
	--learning_rate 1.5e-5 \
	--train_batch_size 64 \
	--cls_drop_out 0.1 "
		;;
	large)
	parameters=" --num_train_epochs 3 \
	--warmup 1000 \
	--learning_rate 1e-5 \
	--train_batch_size 64 \
	--cls_drop_out 0.2 \
	--fp16 True "
		;;
	xlarge)
	parameters=" --num_train_epochs 3 \
	--warmup 1000 \
	--learning_rate 5e-6 \
	--train_batch_size 64 \
	--cls_drop_out 0.3 \
	--fp16 True "
		;;
	xlarge-v2)
	parameters=" --num_train_epochs 3 \
	--warmup 1000 \
	--learning_rate 4e-6 \
	--train_batch_size 64 \
	--cls_drop_out 0.25 \
	--fp16 True "
		;;
	xxlarge-v2)
	parameters=" --num_train_epochs 3 \
	--warmup 1000 \
	--learning_rate 3e-6 \
	--train_batch_size 64 \
	--cls_drop_out 0.3 \
	--fp16 True "
		;;
	xxlarge-v2-sift)
	init=xxlarge-v2
	parameters=" --num_train_epochs 6 \
	--warmup 1000 \
	--vat_lambda 5 \
	--vat_learning_rate 1e-4 \
	--vat_init_perturbation 1e-2 \
	--learning_rate 3e-6 \
	--train_batch_size 64 \
	--cls_drop_out 0.3 \
	--fp16 True "
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

export MASTER_PORT=12456
python -m DeBERTa.apps.run --model_config config.json  \
	--tag $tag \
	--do_train \
	--max_seq_len 256 \
	--eval_batch_size 256 \
	--dump_interval 1000 \
	--task_name $Task \
	--data_dir $cache_dir/glue_tasks/$Task \
	--init_model $init \
	--output_dir /tmp/ttonly/$tag/${task}_v2  $parameters
