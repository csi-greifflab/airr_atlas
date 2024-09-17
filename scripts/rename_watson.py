import csv
import os
DATA_DIR = "/doctorai/userdata/airr_atlas/data/sequences/watson"
OUTPUT_FASTA = f'{DATA_DIR}/watson_combined.fa'
SAMPLES = os.listdir(f'{DATA_DIR}/M')
PREFIX = 'cw'
with open(OUTPUT_FASTA, 'w') as out_f:
    for counter, smpl in enumerate(SAMPLES):
        print(f'Processing {smpl}: {counter+1}/{len(SAMPLES)}')
        igm_dir = f'{DATA_DIR}/M/{smpl}/'
        igg_dir = f'{DATA_DIR}/G/{smpl}/'
        igm_filename = f'{DATA_DIR}/M/{smpl}/{os.listdir(igm_dir)[0]}'
        igg_filename = f'{DATA_DIR}/G/{smpl}/{os.listdir(igg_dir)[0]}'
        
        with open(igm_filename, 'r') as igm:
            with open(igg_filename, 'r') as igg:
                igm_reader = csv.reader(igm, delimiter='\t')
                igg_reader = csv.reader(igg, delimiter='\t')
                for i, igm_row in enumerate(igm_reader):
                    if i == 0:
                        continue
                    id = f'{PREFIX}_{smpl}_IGM_{i}'
                    seq = igm_row[14].replace("*", "")
                    silent = out_f.write(f'>{id}\n{seq}\n')
                for i, igg_row in enumerate(igg_reader):
                    if i == 0:
                        continue
                    id = f'{PREFIX}_{smpl}_IGG_{i}'
                    seq = igg_row[14].replace("*", "")
                    silent = out_f.write(f'>{id}\n{seq}\n')