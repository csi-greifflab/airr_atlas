#!/bin/bash

input_file="/doctorai/marinafr/2023/airr_atlas/analysis/data/all_data/all_data.fa"
output_dir="/doctorai/marinafr/2023/airr_atlas/analysis/output/olga/all_data"

# Number of parallel jobs (adjust as needed)
parallel_jobs=64

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Split the input file into smaller chunks with numbered names
split -d -l $(($(wc -l < "$input_file") / $parallel_jobs)) "$input_file" "$output_dir"/input_chunk_

for ((i=0; i<$parallel_jobs; i++)); do
  chunk_number=$(printf "%02d" $i)  # Ensure two digits, e.g., 00, 01, 02, ...
  olga-compute_pgen --human_B_heavy --infile=${output_dir}/input_chunk_${chunk_number} --outfile=${output_dir}/pgen_H_CDR3_${chunk_number}.fa &
done

# Clean up temporary chunks
#rm input_chunk_*

