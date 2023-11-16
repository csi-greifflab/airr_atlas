#!/bin/bash

source /home/marinafr/.bashrc
conda activate airr_atlas

workers=32

script="/doctorai/marinafr/2023/airr_atlas/analysis/scripts/esm2_bos.py"

input=("/doctorai/marinafr/2023/airr_atlas/analysis/data/all_data/all_data.fa")
output=("/doctorai/marinafr/2023/airr_atlas/analysis/output/esm2/all_data/all_data_bos.pt")

for ((idx=0; idx<${#input[@]}; idx++)); do
    i=${input[$idx]}
    o=${output[$idx]}

    echo "CUDA_VISIBLE_DEVICES=1 python esm2_bos.py $i $o"
    export CUDA_VISIBLE_DEVICES=1
    taskset -c $(mpstat -P ALL 1 1 | awk '$2 ~ /[0-9]/ {print $2, $NF}' | sort -k 2nr | sed '1~2d' | head -n "$workers" | awk '{print $1}' | tr '\n' ',' | sed 's/,$//') python "$script" "$i" "$o"
done
