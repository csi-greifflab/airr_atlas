#!/bin/bash
# NOTE: Paths refer to HPC server. Modify for your environment.

source /home/marinafr/.bashrc
conda activate /doctorai/marinafr/progs/miniconda3/envs/airr_atlas

workers=32

script="/doctorai/marinafr/2023/airr_atlas/analysis/scripts/build_umap.py"

embedding=("antiberta2/freq_test/flanking_res/umap" "esm2/freq_test/flanking_res/umap")

# Nested loops to run the script with different combinations of arguments
for e in "${embedding[@]}"; do
    echo python "$script" "$e" "$workers"
    taskset -c $(mpstat -P ALL 1 1 | awk '$2 ~ /[0-9]/ {print $2, $NF}' | sort -k 2nr | sed '1~2d' | head -n "$workers" | awk '{print $1}' | tr '\n' ',' | sed 's/,$//') python "$script" "$e" 
done
