import csv
import os


#create AIRR format file for trastuzumab heavy chain sequences

input_file = '/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/trastuzumab_binding_labels.csv'
output_file = '/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr.tsv'
repertoire_id = 'tz_heavy'

with open(input_file, 'r') as in_f:
    reader = csv.reader(in_f)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i, row in enumerate(reader):
            if i == 0:
                mute = out_f.write('repertoire_id\tjunction_aa\tbinding_label\n')
                continue
            binding_label = row[4]
            junction_aa = 'CSR' + row[3] + 'YW'
            mute = out_f.write(f'{repertoire_id}\t{junction_aa}\t{binding_label}\n')

# run deduplication with compairr
# call compairr with the following command
compairr_input = output_file
compairr_output = '/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup.tsv'
os.system(f'compairr -zfg {compairr_input} -o {compairr_output}')

duplicates = []
with open(compairr_output, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for i, row in enumerate(reader):
        if i == 0:
            continue
        if row[1] != '1':
            print(row[1])
            duplicates.append(row[2])
        print(f'{i} sequences processed')

# Number of unique sequences = 339483
# Number of duplicates = 5205

# export duplicates to AIRR format file
duplicates_output = '/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_db_final.tsv'
with open(duplicates_output, 'w') as out_f:
    out_f.write('repertoire_id\tsequence_id\tbinding_label\tjunction_aa\n')
    for i, junction_aa in enumerate(duplicates):
        sequence_id = f'tz_heavy_db_{i}'
        binding_label = 'db'
        out_f.write(f'{repertoire_id}_db\t{sequence_id}\t{binding_label}\t{junction_aa}\n')
        print(f'{i} sequences processed')

# remove all duplicates from output_file and save to new file
tz_vh_seq_aa = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSSA"
final_output = '/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv'

with open(output_file, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    with open(final_output, 'w') as out_f:
        counter = 0
        for i, row in enumerate(reader):
            if i == 0:
                out_f.write('repertoire_id\tsequence_id\tbinding_label\tsequence_aa\tcdr3_aa\n')
                continue
            if row[2] not in duplicates:
                counter += 1
                sequence_id = f'{prefix}_{i}'
                binding_label = row[3]
                junction_aa = row[2]
                sequence_aa = tz_vh_seq_aa[:95] + junction_aa + tz_vh_seq_aa[110:]
                out_f.write(f'{repertoire_id}\t{sequence_id}\t{binding_label}\t{sequence_aa}\t{junction_aa}\n')
            print(f'{i} sequences processed')
