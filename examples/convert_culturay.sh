#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

#SBATCH --nodelist=slurm0-a3-ghpc-19
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=160
#SBATCH --mem=1200GB
#SBATCH --job-name=convert_culturay
#SBATCH --output=%x_%j.log

source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate jupyter39

seq -w 0000 3006 | parallel -j 160 echo {} ';' python convert_jsonl_to_parquet.py \
    "culturay_scores_sorted_split_{}.jsonl" \
    "culturay_scores_sorted_split_{}.parquet"
