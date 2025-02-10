#!/bin/bash

startTime=$(date +%s) #mark the start of job 

MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
MASTER_PORT=29400 #5${LSB_JOBID: -5:-1}
NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $HOSTNAME | cut -d':' -f1)-1))
JOB_ID=${LSB_JOBID}

hostname=`hostname`
verbose=0
runtime=$(date "+%Y.%m.%d-%H.%M")
export NVIDIA_PYTORCH_VERSION=3.10 

MACHINE_RANK=$NODE_RANK
MAIN_PROCESS_IP=$MASTER_ADDR
NUM_MACHINES=$NNODES
NUM_PROCESSES=$((NNODES * 8))
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
SEQLEN=4096
LR=2e-06
#LR=5e-06


# fixed precision with allenai env...
#ID="30bmoe_fp_llamafactory"
ID="120bmoe-fp_llamafactory"
MIX_NAME="alpaca_en"

#MODEL_BASE_PATH=/proj/checkpoints/mayank/30b-trial-p2-correct-lr 
MODEL_BASE_PATH=/proj/checkpoints/mayank/120b-trial-p2

NAME=unsharded_model
MODEL_PATH=$MODEL_BASE_PATH/$NAME

OUTPUT_BASE_PATH=/proj/checkpoints/bathen/models/sft/
OUTPUT_MODEL_NAME="${NAME}-${MIX_NAME}-${SEQLEN}-${LR}-${ID}"
OUTPUT_MODEL_PATH="${OUTPUT_BASE_PATH}/${OUTPUT_MODEL_NAME}-model"
DATA_MIX_PATH="${OUTPUT_BASE_PATH}/${OUTPUT_MODEL_NAME}-data"

CODE_PATH=/proj/checkpoints/bathen/developer/LLaMA-Factory

source /u/bathen/run.env

#added to pickup the default conda in /opt/share
. /u/bathen/miniconda3/etc/profile.d/conda.sh
conda_env_path="/proj/checkpoints/bathen/envs/conda/llamafactory"
echo "conda activate ${conda_env_path}"
conda activate ${conda_env_path}
let rc=$?
if [ $rc -ne 0 ]; then
    echo "Conda Activate ${conda_env_path} failed .. Exiting "
    exit $rc
fi

cd $CODE_PATH

export TRITON_CACHE_DIR="/proj/checkpoints/bathen/triton/rank_${NODE_RANK}"
export TRITON_HOME="/proj/checkpoints/bathen/triton/rank_${NODE_RANK}"
#export TORCH_LOGS="recompiles,graph_breaks"
export TORCHINDUCTOR_REORDER_FOR_PEAK_MEMORY=1
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_USE_CUDA_DSA=1
export DISABLE_VERSION_CHECK=1

#weights and biases
#export WANDB_PROJECT=moe-pipelines
export WANDB_DISABLED=1

accelerate launch \
    --mixed_precision bf16 \
    --num_machines $NUM_MACHINES \
    --num_processes $NUM_PROCESSES \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MASTER_PORT \
    --config_file examples/accelerate/fsdp_config.yaml \
    src/train.py examples/granite_moe/granite_moe_sft.yaml


end=$(date +%s)
elapsed=$(($end-$startTime))
echo "Elapsed Time(seconds): ${elapsed}" 

wait


