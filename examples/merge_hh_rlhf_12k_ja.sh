#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

#SBATCH --nodelist=slurm0-a3-ghpc-20
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=160
#SBATCH --mem=1200GB
#SBATCH --job-name=merge
#SBATCH --output=%x_%j.log

source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate jupyter39

dataset_path="llm-jp/hh-rlhf-12k-ja"
dataset_lang="default"
dataset_split="train"
cache_dir="/storage7/askllm/hf_cache"
basename="hh_rlhf_12k_ja_scores_sorted_split"
start=0000
end=0001

seq -w $start $end | parallel -j 160 python merge_askllm.py \
    --input_tsv="${basename}_{}.tsv" \
    --output_jsonl="${basename}_{}.jsonl" \
    --dataset_path="$dataset_path" \
    --dataset_lang="$dataset_lang" \
    --dataset_split="$dataset_split" \
    --cache_dir="$cache_dir"

# check the number of lines in each jsonl file
find . -name "${basename}_????.jsonl" | sort | parallel -k wc -l {}
