#!/bin/bash
#activate ANARCI environment
source /home/jahnz/.bashrc
micromamba activate /doctorai/caters/v_envs/immuneML2

input_dir='/doctorai/userdata/airr_atlas/data/sequences/wang_H_full_chains_batches/'
datasetname='wang_H_full_chains'
output_dir='/doctorai/userdata/airr_atlas/data/sequences/wang_H_full_chains/'
num_cores=24

python /doctorai/userdata/airr_atlas/scripts/parse_anarci_out.py --input_dir=$input_dir --datasetname=$datasetname --output_dir=$output_dir --num_cores=$num_cores