#!/bin/bash
#activate ANARCI environment
source /home/jahnz/.bashrc
micromamba activate /doctorai/jahnz/micromamba/envs/anarci/

input_seqs='/doctorai/userdata/airr_atlas/data/sequences/wang_H_full_chains.fa'
batch_dir='/doctorai/userdata/airr_atlas/data/sequences/wang_H_full_chains_batches/'
cpus=24
batches=24

#split input file into batches
python /doctorai/userdata/airr_atlas/scripts/batch_fasta.py $input_seqs $batch_dir $batches

for file in $batch_dir*.fa; do
    output_csv=${file%.fa}
    
    ANARCI -i $file -s c --csv -o $output_csv --ncpu $cpus --restrict heavy --use_species human
    #replace _H.csv with .csv
    mv ${output_csv}_H.csv $output_csv.csv
    echo "ANARCI progress: $(ls $batch_dir*.csv | wc -l)/$batches"
done

#run ANARCI on each file in batch_dir and report overall progress using GNU parallel
#ls $batch_dir*.fa | parallel --tmpdir=/doctorai/jahnz/tmp -j $cpus ANARCI -i {} -s c --csv -o {.} --ncpu 4 --restrict heavy --use_species human


