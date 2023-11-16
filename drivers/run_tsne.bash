#!/bin/bash

source /home/marinafr/.bashrc
conda activate airr_atlas

workers=40

script="/doctorai/marinafr/2023/airr_atlas/analysis/scripts/build_tsne.py"

embedding=("esm2_tsne" "prott5_tsne")

# Nested loops to run the script with different combinations of arguments
for e in "${embedding[@]}"; do
    echo python "$script" "$e" "$workers"
    taskset -c $(mpstat -P ALL 1 1 | awk '$2 ~ /[0-9]/ {print $2, $NF}' | sort -k 2nr | sed '1~2d' | head -n "$workers" | awk '{print $1}' | tr '\n' ',' | sed 's/,$//') python "$script" "$e" 
done
