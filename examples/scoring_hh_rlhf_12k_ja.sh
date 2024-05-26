#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

#SBATCH --nodelist=slurm0-a3-ghpc-20
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --mem=160GB
#SBATCH --job-name=askllm
#SBATCH --output=%x_%j.log

source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate jupyter39

amount=1500  # 12000 / (8 * 1) = 1500
batch_size=16
max_tokens=4096
output_basename="hh_rlhf_12k_ja_scores"
output_suffix_offset=0  # 0 for node1, 8 for node2, 16 for node3
dataset_path="llm-jp/hh-rlhf-12k-ja"
dataset_lang="default"
dataset_split="train"
dataset_fields="conversations,chosen,rejected"
model_id="Rakuten/RakutenAI-7B-instruct"
cache_dir="/storage7/askllm/hf_cache"
log_interval=100
wandb_project="askllm-test"
wandb_entity="weblab-geniac2"

for rank in {0..7}
do
    python scoring_askllm_for_post_training.py \
        --rank="$rank" \
        --start=$(((output_suffix_offset+rank)*amount)) \
        --end=$(((output_suffix_offset+rank+1)*amount)) \
        --batch_size="$batch_size" \
        --max_tokens="$max_tokens" \
        --output_basename="$output_basename" \
        --output_suffix_offset="$output_suffix_offset" \
        --dataset_path="$dataset_path" \
        --dataset_lang="$dataset_lang" \
        --dataset_split="$dataset_split" \
        --dataset_fields="$dataset_fields" \
        --model_id="$model_id" \
        --cache_dir="$cache_dir" \
        --log_interval="$log_interval" \
        --wandb_project="$wandb_project" \
        --wandb_entity="$wandb_entity" &
done

wait
