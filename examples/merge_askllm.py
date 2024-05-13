# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

import argparse
import sys
import json
from datasets import load_dataset
from tqdm.auto import tqdm


def load_scores(tsv_filename):
    with open(tsv_filename, "r") as f:
        scores = [
            [float(score), int(index)] for (score, index) in [line.strip().split("\t") for line in tqdm(f.readlines())]
        ]
    return scores


def merge_scores(dataset, scores, jsonl_filename):
    with open(jsonl_filename, "w") as f:
        for i, (score, index) in tqdm(enumerate(scores), total=len(scores)):
            data = dataset[index]
            data["askllm_score"] = score
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


parser = argparse.ArgumentParser(description="Merge Ask-LLM scores to the dataset.")
parser.add_argument("--input_tsv", type=str, help="Input tsv file name.", required=True)
parser.add_argument("--output_jsonl", type=str, help="Output jsonl file name.", required=True)
parser.add_argument("--dataset_path", type=str, help="Dataset path name.", default="allenai/c4")
parser.add_argument("--dataset_lang", type=str, help="Dataset language.", default="ja")
parser.add_argument("--dataset_split", type=str, help="Dataset split.", default="train")
parser.add_argument("--cache_dir", type=str, help="Cache directory.", default=None)
args = parser.parse_args()

input_tsv = args.input_tsv
output_jsonl = args.output_jsonl
dataset_path = args.dataset_path
dataset_lang = args.dataset_lang
dataset_split = args.dataset_split
cache_dir = args.cache_dir

print(f"Loading dataset: {dataset_path} {dataset_lang} {dataset_split}", file=sys.stderr)
dataset = load_dataset(dataset_path, name=dataset_lang, split=dataset_split, cache_dir=cache_dir)

print(f"Loading scores: {input_tsv}", file=sys.stderr)
scores = load_scores(input_tsv)

print(f"Merging scores: {output_jsonl}", file=sys.stderr)
merge_scores(dataset, scores, output_jsonl)

print(f"Done: {output_jsonl}", file=sys.stderr)
