#!/bin/bash

SOURCE=$(dirname "$(readlink -f "$0")")/../
if [[ !  -d $SOURCE/DeBERTa ]]; then
  SOURCE=$(dirname "$(readlink -f "$0")")/../../
fi

export PYTHONPATH=${SOURCE}
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
SCRIPT_FILE=$(basename "$SCRIPT")

BashArgs=${@}
DEBUG=False
RESUME=False
Predict=False
ModelType=DeBERTa
ModelSize=Large
INIT_MODEL=""
while [ $# -gt 0 ]; do
  case ${1,,} in
    --debug|-d)
      DEBUG=True
      ;;
    --resume|-r)
      RESUME=True
      ;;
    --init|-i)
      INIT_MODEL=$2
      shift
      ;;
    --predict|-p)
      Predict=True
      ;;
    --output|-o)
      OUTPUT_DIR=$2
      shift
      ;;
    --config|-c)
      CONFIG=$2
      shift
      ;;
    --task|-t)
      Task=$2
      shift
      ;;
    --data)
      Data=$2
      shift
      ;;
    --tag)
      Tag=$2
      shift
      ;;
    --)
      shift
      ExtraArgs=${@}
      break
      ;;
    --help|-h|*)
    echo "Usage $0 [options] -d|--debug  -r|--resume -- <job args>: 
    -d|--debug whether to debug
    -r|--resume whether to resume
      "
      exit 0
      ;;
  esac
  shift
done


export OMP_NUM_THREADS=1

if [[ ${DEBUG,,} = 'true' ]]; then
  export CUDA_VISIBLE_DEVICES=0 #,1 #,1 #,1,2,3,4,5,6,7
fi

export CUDA_VISIBLE_DEVICES=$(python3 -c "import torch; x=[str(x) for x in range(torch.cuda.device_count()) if torch.cuda.get_device_capability(x)[0]>=6]; print(','.join(x))" 2>/dev/null)
IFS=',' read -a DEVICE_CNT <<< "$CUDA_VISIBLE_DEVICES"
MODEL=$INIT_MODEL

if [[ -z $Task ]]; then
  Task=MNLI
fi

if [[ -z $CONFIG ]]; then
	CONFIG=config.json
fi

DUMP=5000
LR_SCH=warmup_linear
CLS_DP=0.15
TAG=${ModelType,,}_${Tag}

if [[ ! -z ${OUTPUT_DIR} ]]; then
  OUTPUT=${OUTPUT_DIR}
else
  OUTPUT=/tmp/job_runs/$Task/$TAG
fi

[ -e $OUTPUT/script ] || mkdir -p $OUTPUT/script

cp -f $CONFIG $OUTPUT/model_config.json

if [[ ! ${Predict,,} = 'true' ]]; then
  CMD=" --do_train"
else
  CMD="--do_eval --do_predict" 
fi

parameters="--task_name $Task $CMD \
  --data_dir $Data \
  --init_model $MODEL \
  --bert_config $OUTPUT/model_config.json \
  --max_seq_length 512 \
  --eval_batch_size 128 \
  --predict_batch_size 128 \
  --output_dir $OUTPUT \
  --scale_steps 250 \
  --loss_scale 16384 \
  --tag $TAG \
  --lr_schedule $LR_SCH \
  --accumulative_update 1 \
  --dump_interval $DUMP \
  --with_radam False \
  --cls_drop_out ${CLS_DP} $ExtraArgs "

python3 -m DeBERTa.apps.train $parameters
