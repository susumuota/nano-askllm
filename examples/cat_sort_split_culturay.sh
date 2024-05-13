#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

cat culturay_scores_?.tsv > culturay_scores_all.tsv
sort -nr culturay_scores_all.tsv > culturay_scores_sorted.tsv
# split -l 3006823 -d -a 4 --additional-suffix=.tsv culturay_scores_sorted.tsv culturay_scores_sorted_split_  # for 10% split
split -l 10000 -d -a 4 --additional-suffix=.tsv culturay_scores_sorted.tsv culturay_scores_sorted_split_      # for 10000 lines split
find . -name "culturay_scores_sorted_split_????.tsv" | sort | parallel -k wc -l {}
