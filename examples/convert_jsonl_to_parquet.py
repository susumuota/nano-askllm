# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

# a naive implementation of jsonl to parquet conversion
# if you have a large file, you may need to implement a more memory-efficient solution

import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


parser = argparse.ArgumentParser(description="Convert jsonl to parquet.")
parser.add_argument("input_jsonl", type=str, help="Input jsonl file name.")
parser.add_argument("output_parquet", type=str, help="Output parquet file name.")
args = parser.parse_args()

input_jsonl = args.input_jsonl
output_parquet = args.output_parquet

df = pd.read_json(input_jsonl, lines=True)
# df = df.drop(columns=[col for col in df.columns if col != "text"])  # keep only the "text" column
table = pa.Table.from_pandas(df)
pq.write_table(table, output_parquet)
