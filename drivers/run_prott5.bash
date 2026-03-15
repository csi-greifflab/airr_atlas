#!/bin/bash
# NOTE: Paths refer to HPC server. Modify for your environment.

#source /home/marinafr/.bashrc
#conda activate airr_atlas

workers=32

script="/doctorai/marinafr/2023/airr_atlas/analysis/scripts/prott5.py"

input=("/doctorai/marinafr/2023/airr_atlas/analysis/data/ireceptor/H_CDR3.fa")
output=("/doctorai/marinafr/2023/airr_atlas/analysis/output/prott5/ireceptor/H_CDR3.pkl")

for ((idx=0; idx<${#input[@]}; idx++)); do
    i=${input[$idx]}
    o=${output[$idx]}

    echo "CUDA_VISIBLE_DEVICES=1 python prott5.py $i $o"
    export CUDA_VISIBLE_DEVICES=0
    taskset -c $(mpstat -P ALL 1 1 | awk '$2 ~ /[0-9]/ {print $2, $NF}' | sort -k 2nr | sed '1~2d' | head -n "$workers" | awk '{print $1}' | tr '\n' ',' | sed 's/,$//') python "$script" "$i" "$o"
done
