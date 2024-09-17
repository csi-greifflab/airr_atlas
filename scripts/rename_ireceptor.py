import os
from Bio import SeqIO
def add_prefix_fasta_records(input_file, fasta_out, airr_out, prefix, repertoire_id):
    """
    Adds a prefix to the sequence IDs in a fasta file. Prints progress and estimated time remaining.

    Args:
        input_file (str): Path to the input fasta file.
        output_file (str): Path to the output fasta file.
        prefix (str): Prefix to add to the sequence IDs.
    """
    print('Adding prefix to sequence IDs...')
    with open(input_file, 'r') as f:
        with open(fasta_out, 'a') as out_f:
            with open(airr_out, 'a') as airr_f:
                airr_f.write('repertoire_id\tsequence_id\tjunction_aa\n')
                for i, record in enumerate(SeqIO.parse(f, 'fasta')):
                    record.id = f'{prefix}{i}'
                    record.description = ''
                    SeqIO.write(record, out_f, 'fasta')
                    airr_f.write(f'{repertoire_id}\t{record.id}\t{record.seq}\n')
                    print(f'{i+1} sequences processed', end='\r')

input_file = '/doctorai/userdata/airr_atlas/data/sequences/ireceptor/ireceptor_H_cdr3.fa'
fasta_out = '/doctorai/userdata/airr_atlas/data/sequences/ireceptor/ireceptor_H_cdr3_renamed.fa'
airr_out = '/doctorai/userdata/airr_atlas/data/sequences/ireceptor/ireceptor_H_cdr3.tsv'
prefix = '02'
repertoire_id = 'irec_sub'
add_prefix_fasta_records(input_file, fasta_out, airr_out, prefix, repertoire_id)
