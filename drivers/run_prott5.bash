#!/bin/bash

source /home/marinafr/.bashrc
conda activate airr_atlas

workers=32

script="/doctorai/marinafr/2023/airr_atlas/analysis/scripts/prott5.py"

chain=("H")
input=("CDR3")

# Nested loops to run the script with different combinations of arguments
for c in "${chain[@]}"; do
    for i in "${input[@]}"; do
        echo CUDA_VISIBLE_DEVICES=1 python "$script" "$c" "$i"
        export CUDA_VISIBLE_DEVICES=1
        taskset -c $(mpstat -P ALL 1 1 | awk '$2 ~ /[0-9]/ {print $2, $NF}' | sort -k 2nr | sed '1~2d' | head -n "$workers" | awk '{print $1}' | tr '\n' ',' | sed 's/,$//') python "$script" "$c" "$i"
    done
done
