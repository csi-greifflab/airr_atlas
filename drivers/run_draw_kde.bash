#!/bin/bash

source /home/marinafr/.bashrc
conda activate airr_atlas

workers=40

script="/doctorai/marinafr/2023/airr_atlas/analysis/scripts/draw_kde.py"

input=("/doctorai/marinafr/2023/airr_atlas/analysis/output/esm2/all_data/")

echo python "$script" "$input" "$workers"
taskset -c $(mpstat -P ALL 1 1 | awk '$2 ~ /[0-9]/ {print $2, $NF}' | sort -k 2nr | sed '1~2d' | head -n "$workers" | awk '{print $1}' | tr '\n' ',' | sed 's/,$//') python "$script" "$input" 
