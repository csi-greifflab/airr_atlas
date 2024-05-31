#!/bin/bash

source /home/jahnz/.bashrc
conda activate /doctorai/jahnz/micromamba/envs/esm2

workers=1

script="/doctorai/userdata/airr_atlas/scripts/esm2_cdr3.py"

input=("/doctorai/userdata/airr_atlas/data/sequences/first_10.fasta")
output=("test.pt")
cdr3_file=("/doctorai/userdata/airr_atlas/data/sequences/wang_H_full_chains/wang_H_full_chains_cdr3.csv")

# TODO add layers and context parameter

for ((idx=0; idx<${#input[@]}; idx++)); do
    i=${input[$idx]}
    o=${output[$idx]}

    export CUDA_VISIBLE_DEVICES=0
    taskset -c $(mpstat -P ALL 1 1 | awk '$2 ~ /[0-9]/ {print $2, $NF}' | sort -k 2nr | sed '1~2d' | head -n "$workers" | awk '{print $1}' | tr '\n' ',' | sed 's/,$//') python "$script" "$i" "$o" "--cdr3_path $cdr3_file"
done
