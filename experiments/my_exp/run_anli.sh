#!/bin/bash
#
# This is an example script to show how to made customized task
#
#
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR


Task=AlphaNLI
# Download the data from https://leaderboard.allenai.org/anli/submissions/get-started

init=$1
tag=$init
case ${init,,} in
	base-mnli)
	parameters=" --num_train_epochs 6 \
	--warmup 1000 \
	--learning_rate 2e-5 \
	--train_batch_size 64 \
	--cls_drop_out 0.2 \
	--max_seq_len 70"
		;;
	large-mnli)
	parameters=" --num_train_epochs 6 \
	--warmup 1000 \
	--learning_rate 8e-6 \
	--train_batch_size 64 \
	--cls_drop_out 0.2 \
	--max_seq_len 70"
		;;
	xlarge-mnli)
	parameters=" --num_train_epochs 6 \
	--warmup 1000 \
	--learning_rate 7e-6 \
	--train_batch_size 64 \
	--cls_drop_out 0.2\
	--fp16 True \
	--max_seq_len 70"
		;;
	xlarge-v2-mnli)
	parameters=" --num_train_epochs 6 \
	--warmup 1000 \
	--learning_rate 4e-6 \
	--train_batch_size 128 \
	--cls_drop_out 0.2 \
	--fp16 True \
	--max_seq_len 70"
		;;
	xxlarge-v2-mnli)
	parameters=" --num_train_epochs 6 \
	--warmup 500 \
	--learning_rate 8e-6 \
	--train_batch_size 256 \
	--cls_drop_out 0.1 \
	--fp16 True \
	--max_seq_len 70"
		;;
	eval)
	init=anli_models/avg.bin
	vocab=anli_models/spm.model
	
	parameters=" --num_train_epochs 6 \
	--warmup 500 \
	--vocab_type spm \
	--vocab_path $vocab \
	--learning_rate 8e-6 \
	--train_batch_size 256 \
	--cls_drop_out 0.1 \
	--fp16 True \
	--max_seq_len 70"
		;;
	deberta-v3-large)
	parameters=" --num_train_epochs 10 \
	--warmup 1000 \
	--learning_rate 7e-6 \
	--train_batch_size 256 \
	--cls_drop_out 0.1 \
	--fp16 True \
	--max_seq_len 70"
		;;
	*)
		echo "usage $0 <Pretrained model configuration>"
		echo "Supported configurations"
		echo "base-mnli - Pretrained DeBERTa v1 model with 140M parameters (12 layers, 768 hidden size)"
		echo "large-mnli - Pretrained DeBERta v1 model with 380M parameters (24 layers, 1024 hidden size)"
		echo "xlarge-mnli - Pretrained DeBERTa v1 model with 750M parameters (48 layers, 1024 hidden size)"
		echo "xlarge-v2-mnli - Pretrained DeBERTa v2 model with 900M parameters (24 layers, 1536 hidden size)"
		echo "xxlarge-v2-mnli - Pretrained DeBERTa v2 model with 1.5B parameters (48 layers, 1536 hidden size)"
		echo "deberta-v3-large - Pretrained DeBERTa v3 large model with 480M parameters (24 layers, 1024 hidden size)"
		exit 0
		;;
esac

python  -W ignore  -m DeBERTa.apps.run --model_config config.json  \
	--tag $tag \
	--do_train \
	--task_dir . \
	--task_name $Task \
	--init_model $init \
	--data_dir /mount/biglm_data/alphanli \
	--output_dir /tmp/ttonly/$tag/$Task  $parameters
	#--do_eval \
	#--do_predict \
