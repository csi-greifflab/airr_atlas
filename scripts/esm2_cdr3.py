"""
This script takes a fasta file as input and outputs a tensor of the mean representations of each sequence in the file using a pre-trained ESM-2 model.
The tensor is saved as a PyTorch file.

Args:
    fasta_path (str): Path to the fasta file.
    output_path (str): Path to save the output tensor.
"""
import torch
from esm import FastaBatchedDataset, pretrained
import argparse
import numpy as np
from Bio import SeqIO
import pandas as pd


# Parsing command-line arguments for input and output file paths
parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("fasta_path", type=str, help="Fasta path + filename.fa")
parser.add_argument("output_path", type=str, help="Output path + filename.pt")
parser.add_argument("--cdr3_path", default = None, type=str, help="Path to the CDR3 CSV file. Only required when calculating CDR3 sequence embeddings.")
parser.add_argument("--context", default = 0,type=int, help="Number of amino acids to include before and after CDR3 sequence")
parser.add_argument("--layers", default = [-1],type=int, help="Representation layers to extract from the model. Default is the last layer.")
args = parser.parse_args()

# Storing the input and output file paths
fasta_file = args.fasta_path
output_file = args.output_path
cdr3_path = args.cdr3_path
context = args.context
layers = args.layers

#debug values
#fasta_file = '/doctorai/userdata/airr_atlas/data/sequences/first_10.fasta'
#output_file = '/doctorai/userdata/airr_atlas/test_cdr3.pt'
#cdr3_path = '/doctorai/userdata/airr_atlas/data/sequences/wang_H_full_chains/wang_H_full_chains_cdr3.csv'
#context = 0


cdr3_df = pd.read_csv(cdr3_path)


#convert fasta into pandas dataframe
def fasta_to_df(fasta_file):
    fasta_sequences = SeqIO.parse(open(fasta_file),'fasta')
    df = pd.DataFrame(columns=['id', 'sequence'])
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        df = pd.concat([df, pd.DataFrame({'id': [name], 'sequence': [sequence]})], ignore_index=True)
    return df

fasta_sequences = fasta_to_df(fasta_file)

# Pre-defined model location and batch token size
MODEL_LOCATION = "esm2_t33_650M_UR50D"
TOKS_PER_BATCH = 4096
REPR_LAYERS = layers

# Loading the pretrained model and alphabet for tokenization
model, alphabet = pretrained.load_model_and_alphabet(MODEL_LOCATION)
model.eval()  # Setting the model to evaluation mode

# Moving the model to GPU if available for faster processing
if torch.cuda.is_available():
    model = model.cuda()
    print("Transferred model to GPU")

# Creating a dataset from the input fasta file
dataset = FastaBatchedDataset.from_file(fasta_file)
# Generating batch indices based on token count
batches = dataset.get_batch_indices(TOKS_PER_BATCH, extra_toks_per_seq=1)
# DataLoader to iterate through batches efficiently
data_loader = torch.utils.data.DataLoader(
    dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
)

print(f"Read {fasta_file} with {len(dataset)} sequences")

# Checking if the specified representation layers are valid
assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in REPR_LAYERS)
repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in REPR_LAYERS]

# Initializing lists to store mean representations and sequence labels
mean_representations = {layer: [] for layer in repr_layers}
seq_labels = []

# Processing each batch without computing gradients (to save memory and computation)
with torch.no_grad():
    for batch_idx, (labels, strs, toks) in enumerate(data_loader):
        print(
            f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
        )
        # Moving tokens to GPU if available
        if torch.cuda.is_available():
            toks = toks.to(device="cuda", non_blocking=True)

        # Computing representations for the specified layers
        out = model(toks, repr_layers=repr_layers, return_contacts=False)

        # Extracting layer representations and moving them to CPU
        representations = {
            layer: t.to(device="cpu") for layer, t in out["representations"].items()
        }
        
        # Mean pooling representations for each sequence, excluding the beginning-of-sequence (bos) token
        for i, label in enumerate(labels):
            try:
                cdr3_sequence = cdr3_df[cdr3_df['id'] == label]['cdr3_sequence'].values[0]
            except:
                print(f'No cdr3 sequence found for {label}')
                continue
            full_sequence = fasta_sequences[fasta_sequences['id'] == label]['sequence'].values[0] 
            # remove '-' from cdr3_sequence
            cdr3_sequence = cdr3_sequence.replace('-', '')

            # get position of cdr3_sequence in sequence
            start = full_sequence.find(cdr3_sequence) - context
            end = start + len(cdr3_sequence) + context
            seq_labels.append(label)
            for layer in representations.keys():
                mean_representation = representations[layer][i, start : end].mean(0).clone()
                # We take mean_representation[0] to keep the [array] instead of [[array]].
                mean_representations[layer].append(mean_representation)

            #mean_representation = [t[i, start : end].mean(0).clone()
            #        for layer, t in representations.items()]



# Sorting the representations based on sequence labels
ordering = np.argsort([int(i) for i in seq_labels])

# Stacking all mean representations into a single tensor and save to output file
for layer in mean_representations.keys():
    mean_representations[layer] = torch.vstack(mean_representations[layer])
    mean_representations[layer] = mean_representations[layer][ordering, :]

    output_file = output_file.replace('.pt', f'_layer_{layer}.pt')
    torch.save(mean_representations[layer], output_file)
    