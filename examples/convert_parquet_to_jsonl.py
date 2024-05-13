# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

# a naive implementation of parquet to jsonl conversion
# if you have a large file, you may need to implement a more memory-efficient solution

import argparse
import pandas as pd


parser = argparse.ArgumentParser(description="Convert parquet to jsonl.")
parser.add_argument("input_parquet", type=str, help="Input parquet file name.")
parser.add_argument("output_jsonl", type=str, help="Output jsonl file name.")
args = parser.parse_args()

input_parquet = args.input_parquet
output_jsonl = args.output_jsonl

df = pd.read_parquet(input_parquet)
df.to_json(output_jsonl, orient='records', lines=True, force_ascii=False)
