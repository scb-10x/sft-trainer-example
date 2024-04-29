#!/bin/bash
#SBATCH -p gpu                # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16             # Specify number of nodes and processors per task
#SBATCH --gpus=1              # Specify total number of GPUs
#SBATCH --ntasks-per-node=1   # Specify tasks per node
#SBATCH -t 120:00:00          # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt900047           # Specify project name
#SBATCH -J JOBNAME            # Specify job name

ml Mamba
conda deactivate
conda activate /project/lt900048-ai24tn/scb10x_env # Activate selected environment
python train.py --dataset_name /project/lt900048-ai24tn/datasets/scb10x/scb_mt_enth_2020_aqdf_1k/scb_mt_enth_2020_aqdf_1k_train.jsonl --model_name /project/lt900048-ai24tn/models/scb10x/typhoon-7b --gradient_accumulation_steps 4