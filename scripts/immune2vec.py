##-----------------------------
## 
## Script name: train_immune2vec.py
##
## Purpose of script: Script to train immune2vec on four single-cell BCR datasets
##
## Author: Mamie Wang
##
## Date Created: 2022-08-18
##
## Email: mamie.wang@yale.edu
##
## ---------------------------
## load the packages and inputs

import pickle
import sys
import os

#sys.path.append("/gpfs/ysm/project/mw957/repos/archive/immune2vec/embedding")
sys.path.append("/doctorai/marinafr/2023/airr_atlas/wang_paper/libraries/")
from immune2vec_model.embedding import sequence_modeling

import pandas as pd
import numpy as np
from Bio import SeqIO
import argparse


parser = argparse.ArgumentParser(description="Gene usage tasks")
parser.add_argument("input_csv", type=str, help="Input path + filename.csv")
parser.add_argument("output_path", type=str, help="Output path")
parser.add_argument("dim", type=int, help="Latent dimesnion of immune2vec")
parser.add_argument("workers", type=int, help="n_threads")
args = parser.parse_args()

##-----------------------------
## Run immune2vec

sequence_train = pd.read_csv(args.input_csv, sep=';').junction_aa
out_dir = args.output_path
out_corpus_fname = args.output_path + 'all_data'

seq_len = sequence_train.apply(len)
print("# sequences: " + str(sequence_train.shape[0]))

lengths = sequence_train.apply(len)
print("Length range: " + str(min(lengths)) + " - " + str(max(lengths)) + " AA.")


def generate_model_exec(sequences, 
                       output_folder_path,
                       desc,
                       n_dim=args.dim,
                       data_fraction=1.0,
                       seed=0,
                       n_gram=3,
                       reading_frame=None):

    pv = sequence_modeling.ProtVec(data=sequences, corpus=os.path.join(output_folder_path, desc),
                                   n=n_gram, reading_frame=reading_frame,
                                   size=n_dim, out=desc, sg=1, window=5, min_count=2, workers=args.workers,
                                   sample_fraction=data_fraction, random_seed=seed)

    print('Model is ready, saving ' + desc + ".immune2vec")
    pv.save(desc + ".immune2vec")

generate_model_exec(sequences=sequence_train,
                output_folder_path=out_dir,
                desc=out_corpus_fname)

##-----------------------------
## Embed

# Load fasta file
#ids = []
#seqs = []
#for seq_record in SeqIO.parse(fasta_file, "fasta"):
#    ids.append(seq_record.id)
#    seqs.append(''.join(seq_record.seq))
    
#seqs = pd.Series(seqs, index = ids)
#print(f"Read {fasta_file} with {len(seqs)} sequences")

def generate_vectors_exec(out_fname, 
                          model_path, 
                          n_read_frames=3):

    data_file = sequence_train

    # load saved model
    model = sequence_modeling.load_protvec(model_path)

    # generate a vector for each junction
    data_len = len(data_file)
    print('Data length: ' + str(data_len))

    def embed_data(word):
        try:
            return list(model.to_vecs(word, n_read_frames=n_read_frames))
        except:
            return np.nan

    print('Generating vectors...')
    vectors = data_file.apply(embed_data)

    print('{:.3}% of data not transformed'.format((100*sum(vectors.isna())/data_len)))

    # drop the un translated rows from the file
    data_file = data_file.drop(vectors[vectors.isna()].index, axis=0)
    vectors = vectors[vectors.notna()]

    # save to files:
    data_file_output = os.path.join(out_fname + '_filtered.csv')
    print('Saving ' + data_file_output)
    data_file.to_csv(data_file_output, sep=';', index=False)

    vectors = np.array(vectors.tolist())

    vectors_file_output = os.path.join(out_fname + '_vectors.npy')
    print('Saving ' + vectors_file_output)
    np.save(vectors_file_output, vectors)
    
generate_vectors_exec(out_fname=out_corpus_fname,
                      model_path=out_corpus_fname + ".immune2vec")
