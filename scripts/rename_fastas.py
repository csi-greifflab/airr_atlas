import os
from Bio import SeqIO
def add_prefix_fasta_records(input_file, output_file, prefix):
    """
    Adds a prefix to the sequence IDs in a fasta file. Prints progress and estimated time remaining.

    Args:
        input_file (str): Path to the input fasta file.
        output_file (str): Path to the output fasta file.
        prefix (str): Prefix to add to the sequence IDs.
    """
    print('Adding prefix to sequence IDs...')
    with open(input_file, 'r') as f:
        with open(output_file, 'a') as out_f:
            for i, record in enumerate(SeqIO.parse(f, 'fasta')):
                record.id = f'{prefix}_{record.id}'
                record.description = ''
                SeqIO.write(record, out_f, 'fasta')
                print(f'{i+1} sequences processed', end='\r')

#input_file = '/doctorai/userdata/airr_atlas/data/sequences/wang/wang_H_full_chains.fa'
#output_file = '/doctorai/userdata/airr_atlas/data/sequences/wang/wang_H_full_chains_renamed.fa'
#prefix = 'wang_heavy'
#add_prefix_fasta_records(input_file, output_file, prefix)

#input_file = '/doctorai/userdata/airr_atlas/data/sequences/antigens/covabdab/covabdab.fa'
#output_file = '/doctorai/userdata/airr_atlas/data/sequences/antigens/covabdab/covabdab_renamed.fa'
#prefix = 'cov'
#add_prefix_fasta_records(input_file, output_file, prefix)

#input_file = '/doctorai/userdata/airr_atlas/data/sequences/antigens/antigens/influenza/influenza_full_chains.fa'
#output_file = '/doctorai/userdata/airr_atlas/data/sequences/antigens/antigens/influenza/influenza_full_chains_renamed.fa'
#prefix = 'inf'
#add_prefix_fasta_records(input_file, output_file, prefix)

#input_file = '/doctorai/userdata/airr_atlas/data/sequences/briney/briney_900k.fa'
#output_file = '/doctorai/userdata/airr_atlas/data/sequences/briney/briney_900k_renamed.fa'
#prefix = 'briney_sub'
#add_prefix_fasta_records(input_file, output_file, prefix)

input_file = '/doctorai/userdata/airr_atlas/data/sequences/antigens/antigens/ebola/ebola_specific.fa'
output_file = '/doctorai/userdata/airr_atlas/data/sequences/antigens/antigens/ebola/ebola_specific_renamed.fa'
prefix = 'ebola'
add_prefix_fasta_records(input_file, output_file, prefix)
