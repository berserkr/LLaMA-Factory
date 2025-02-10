#!/bin/bash

#LOG=fspd-30b-llamafactory-sft
LOG=fspd-120b-llamafactory-sft

#NNODES=8
NNODES=24

CODEBASE=/proj/checkpoints/bathen/developer/LLaMA-Factory
SCRIPT=examples/granite_moe/train.sh 

bsub -U p1345nodes -M 1024G -P "challenge:6430" -gpu "num=8/task:mode=exclusive_process" -n $NNODES -o logs/${LOG}.out -e logs/${LOG}.err blaunch $CODEBASE/$SCRIPT
