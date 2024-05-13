#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

#SBATCH --nodelist=slurm0-a3-ghpc-19
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=160
#SBATCH --mem=1200GB
#SBATCH --job-name=merge_culturay
#SBATCH --output=%x_%j.log

source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate jupyter39

dataset_path="ontocord/CulturaY"
dataset_lang="ja"
dataset_split="train"
cache_dir="/storage7/askllm/hf_cache"

seq -w 0000 3006 | parallel -j 160 python merge_askllm.py \
    --input_tsv="culturay_scores_sorted_split_{}.tsv" \
    --output_jsonl="culturay_scores_sorted_split_{}.jsonl" \
    --dataset_path="$dataset_path" \
    --dataset_lang="$dataset_lang" \
    --dataset_split="$dataset_split" \
    --cache_dir="$cache_dir"

# check the number of lines in each jsonl file
find . -name "culturay_scores_sorted_split_????.jsonl" | sort | parallel -k wc -l {}
