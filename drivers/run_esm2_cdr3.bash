#!/bin/bash

source /home/jahnz/.bashrc
conda activate /doctorai/jahnz/micromamba/envs/esm2

script="/doctorai/userdata/airr_atlas/scripts/esm2_cdr3.py"

input=("/doctorai/userdata/airr_atlas/data/sequences/wang_H_full_chains.fa")
output=("/doctorai/userdata/airr_atlas/data/embeddings/wang_H_full_chains_cdr3.pt")
cdr3_file=("/doctorai/userdata/airr_atlas/data/sequences/wang_H_full_chains/wang_H_full_chains_cdr3.csv")
layers=("6 33")
# TODO add context parameter

export CUDA_VISIBLE_DEVICES=0
python "$script" --fasta_path "$input" --output_path "$output" --cdr3_path "$cdr3_file" --layers "$layers"
#python "$script" "--fasta_path $input --output_path $output --cdr3_path $cdr3_file --layers $layers"

