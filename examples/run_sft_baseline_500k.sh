#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

#SBATCH --nodelist=slurm0-a3-ghpc-0
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160GB
#SBATCH --job-name=askllm
#SBATCH --output=%x_%j.log

source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate askllm311

start=0
end=-1
batch_size=8
max_tokens=4096
output_tsv="output_train.tsv"
# output_tsv="output_eval.tsv"
dataset_path="team-hatakeyama-phase2/sft_baseline_500k"
dataset_lang="default"
dataset_split="train"
# dataset_split="eval"
dataset_fields="instruction,input,output"
# model_id="Rakuten/RakutenAI-7B-instruct"
model_id="microsoft/Phi-3-medium-128k-instruct"
cache_dir="/storage5/shared/ota_s/askllm/hf_cache"
log_interval=100
wandb_project="askllm"
wandb_entity="weblab-geniac1"
wandb_name="sft_baseline_500k_train"
# wandb_name="sft_baseline_500k_eval"

rm -f "$output_tsv"
time python run_askllm.py \
    --start="$start" \
    --end="$end" \
    --batch_size="$batch_size" \
    --max_tokens="$max_tokens" \
    --output_tsv="$output_tsv" \
    --dataset_path="$dataset_path" \
    --dataset_lang="$dataset_lang" \
    --dataset_split="$dataset_split" \
    --dataset_fields="$dataset_fields" \
    --model_id="$model_id" \
    --cache_dir="$cache_dir" \
    --log_interval="$log_interval" \
    --wandb_project="$wandb_project" \
    --wandb_entity="$wandb_entity" \
    --wandb_name="$wandb_name"

output_sorted_tsv=$(basename $output_tsv .tsv)_sorted.tsv
rm -f "$output_sorted_tsv"
sort -nr "$output_tsv" > "$output_sorted_tsv"

output_jsonl=$(basename $output_tsv .tsv).jsonl
rm -f "$output_jsonl"
time python merge_askllm.py \
    --input_tsv="$output_sorted_tsv" \
    --output_jsonl="$output_jsonl" \
    --dataset_path="$dataset_path" \
    --dataset_lang="$dataset_lang" \
    --dataset_split="$dataset_split" \
    --cache_dir="$cache_dir"
