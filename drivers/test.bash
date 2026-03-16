#!/bin/bash

input_file="/doctorai/marinafr/2023/airr_atlas/analysis/data/all_data/all_data.fa"
output_dir="/doctorai/marinafr/2023/airr_atlas/analysis/output/olga/all_data"

olga-compute_pgen --human_B_heavy --infile=${output_dir}/input_chunk_64 --outfile=${output_dir}/pgen_H_CDR3_64.fa

