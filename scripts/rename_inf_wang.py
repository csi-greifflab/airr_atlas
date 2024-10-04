import csv

input_file = '/doctorai/userdata/airr_atlas/data/sequences/antigens/antigens/alphaseq_sars/alphaseq_df.csv'
output_file = '/doctorai/userdata/airr_atlas/data/sequences/antigens/antigens/alphaseq_sars/alphaseq_df_renamed.csv'
output_fasta = '/doctorai/userdata/airr_atlas/data/sequences/antigens/antigens/alphaseq_sars/alphaseq_H_full_renamed.fa'
prefix = 'alphaseq'
with open(input_file, 'r') as in_f:
    reader = csv.reader(in_f)
    with open(output_file, 'w') as out_f:
        with open(output_fasta, 'w') as fasta_f:
            for i, row in enumerate(reader):
                if i == 0:
                    mute = out_f.write(','.join(row)+'\n')
                    continue
                row[0] = f'{prefix}_{i}'
                mute = out_f.write(','.join(row)+'\n')
                mute = fasta_f.write(f'>{row[0]}\n{row[6]}\n')