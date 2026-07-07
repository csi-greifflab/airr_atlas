#!/bin/bash
# NOTE: Paths refer to HPC server. Modify for your environment.

#source /home/marinafr/.bashrc
#conda activate airr_atlas

workers=32

script="/doctorai/marinafr/2023/airr_atlas/analysis/scripts/antiberta2_cdr3.py"

input=(
"/doctorai/marinafr/2023/airr_atlas/analysis/data/ireceptor/random/random_aa.fa" "/doctorai/marinafr/2023/airr_atlas/analysis/data/ireceptor/random/random_shuffled.fa" "/doctorai/marinafr/2023/airr_atlas/analysis/data/ireceptor/random/random.fa"
)

output=(
"/doctorai/marinafr/2023/airr_atlas/analysis/output/antiberta2/ireceptor/random/random_aa.pt" "/doctorai/marinafr/2023/airr_atlas/analysis/output/antiberta2/ireceptor/random/random_shuffled.pt" "/doctorai/marinafr/2023/airr_atlas/analysis/output/antiberta2/ireceptor/random/random.pt"
)

for ((idx=0; idx<${#input[@]}; idx++)); do
    f=${input[$idx]}
    o=${output[$idx]}

    echo "CUDA_VISIBLE_DEVICES=1 python esm2.py $f $o $c"
    export CUDA_VISIBLE_DEVICES=0
    taskset -c $(mpstat -P ALL 1 1 | awk '$2 ~ /[0-9]/ {print $2, $NF}' | sort -k 2nr | sed '1~2d' | head -n "$workers" | awk '{print $1}' | tr '\n' ',' | sed 's/,$//') python "$script" --fasta_path "$f" --output_path "$o" --layers -1
done
