"""
This script takes a fasta file as input and outputs a tensor of the mean representations of each sequence in the file using a pre-trained ESM-2 model.
The tensor is saved as a PyTorch file.

Args:
    fasta_path (str): Path to the fasta file.
    output_path (str): Path to save the output tensor.
"""

import torch
from esm import FastaBatchedDataset
from transformers import RoFormerTokenizer, RoFormerModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from Bio import SeqIO
import os
import csv


#Parsing command-line arguments for input and output file paths
parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("--fasta_path", type=str, required=True,
                    help="Fasta path + filename.fa")
parser.add_argument("--output_path", type=str, required=True,
                    help="Output path + filename.pt \nWill output multiple files if multiple layers are specified with '--layers'.")
parser.add_argument("--cdr3_path", default = None, type=str,
                    help="Path to the CDR3 CSV file. Only required when calculating CDR3 sequence embeddings.")
parser.add_argument("--context", default = 0,type=int,
                    help="Number of amino acids to include before and after CDR3 sequence")
parser.add_argument("--layers", type=str, nargs='*', default="-1",
                    help="Representation layers to extract from the model. Default is the last layer. Example: argument '--layers -1 6' will output the last layer and the sixth layer.")

args = parser.parse_args()

# Storing arguments
fasta_path = args.fasta_path
output_path = args.output_path
cdr3_path = args.cdr3_path
context = args.context
layers = list(map(int, args.layers[0].split()))

########test
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#fasta_path = '/doctorai/userdata/airr_atlas/data/sequences/wang_H_full_chains.fa'
#output_path = '/doctorai/userdata/airr_atlas/test/test_cdr3.pt'
#cdr3_path = '/doctorai/userdata/airr_atlas/data/sequences/wang_H_full_chains/wang_H_full_chains_cdr3.csv'
#context = 0
#layers = list(range(1,17))

# Load cdr3 sequences and store in dictionary
with open(cdr3_path) as f:
    reader = csv.reader(f)
    cdr3_dict = {rows[0]:rows[1] for rows in reader}
#######
# convert fasta into dictionary
def fasta_to_dict(fasta_file):
    print('Loading and batching input sequences...')
    seq_dict = {}
    with open(fasta_file) as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq_dict[record.id] = " ".join(str(record.seq)) # AA tokens for hugging face models must be gapped
            # print progress
            if len(seq_dict) % 1000 == 0:
                return seq_dict
                print(f'{len(seq_dict)} sequences loaded')
    return seq_dict

# Read sequences from the FASTA file
sequences = fasta_to_dict(fasta_path)

# TODO investigate missing_keys
missing_keys = [key for key in sequences.keys() if key not in cdr3_dict.keys()]


# Pre-defined parameters for optimization
MODEL_NAME = "alchemab/antiberta2-cssp"
BATCH_SIZE = 4096  # Adjust based on your GPU's memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


tokenizer = RoFormerTokenizer.from_pretrained(MODEL_NAME)
model = RoFormerModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


print("Start tokenization")

# Tokenize sequences
input_ids = []
attention_masks = []
total_sequences = len(sequences)
counter = 0
for sequence in sequences.values():
    counter += 1
    tokens = tokenizer(sequence,truncation=True, padding='max_length', return_tensors="pt",add_special_tokens=True, max_length=200)
    #print( tokenizer.decode(tokens['input_ids'][0]))
    input_ids.append(tokens['input_ids'])
    attention_masks.append(tokens['attention_mask'])
    # Calculate and print the percentage of completion
    percent_complete = ((counter + 1) / total_sequences) * 100
    # Check and print the progress at each 2% interval
    if (counter + 1) == total_sequences or int(percent_complete) % 2 == 0:
        # Ensures the message is printed once per interval and at 100% completion
        if (counter + 1) == total_sequences or (int(percent_complete / 2) != int(((counter) / total_sequences) * 100 / 2)):
            print(f"Progress: {percent_complete:.2f}%")



# Convert lists to tensors and create a dataset
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
dataset = TensorDataset(input_ids, attention_masks)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Initialize a list to store embeddings
mean_representations = {layer: [] for layer in layers}
seq_labels = []
with torch.no_grad():
    total_batches = len(data_loader)  # Correctly calculate the total number of batches here
    for (batch_idx, batch), labels in zip(enumerate(data_loader), sequences.keys()):
        input_ids, attention_mask = [b.to(device, non_blocking=True) for b in batch]
        #print(input_ids[0])
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True)
        # Extracting layer representations and moving them to CPU
        representations = {}
        for layer in layers:
            representations[layer] = outputs.hidden_states[layer].to(device="cpu")
        
        # TODO add optional argument to return mean pooled full embedding even if cdr3_path is specified
        if cdr3_path is None:
            for layer in layers:
                for counter, label in enumerate(labels):
                    mean_representation = representations[layer][counter, len(sequences[counter]) + 1].mean(0).clone()
                    # We take mean_representation[0] to keep the [array] instead of [[array]].
                    mean_representations[layer].append(mean_representation)
        else:
            for counter, label in enumerate(labels):
                try:
                    cdr3_sequence = cdr3_dict[label]
                except:
                    if label not in missing_keys:
                        print(f'No cdr3 sequence found for {label}')
                    continue
                print(f'Processing {label}')

                # load sequence without gaps
                full_sequence = sequences[label].replace(' ', '')

                # remove '-' from cdr3_sequence
                cdr3_sequence = cdr3_sequence.replace('-', '')

                # get position of cdr3_sequence in sequence
                try:
                    start = full_sequence.find(cdr3_sequence) - context
                except:
                    print("Context window too large")
                try:
                    end = start + len(cdr3_sequence) + context
                except:
                    print("Context window too large")
                seq_labels.append(label)

                for layer in layers:
                    mean_representation = representations[layer][counter, start : end].mean(0).clone()
                    # We take mean_representation[0] to keep the [array] instead of [[array]].
                    mean_representations[layer].append(mean_representation)
        

        # #print(embeddings)
        # first_token_batch = embeddings[:, 0, :]
        # #print(first_token_batch)
        # all_embeddings.append(first_token_batch.cpu())
        # Correctly print the progress
        print(
            f"Processing {batch_idx + 1} of {total_batches} batches ({input_ids.size(0)} sequences)"
        )


# Sorting the representations based on sequence labels
ordering = np.argsort([int(i) for i in seq_labels])

# Stacking representations of each layer into a single tensor and save to output file
for layer in layers:
    mean_representations[layer] = torch.vstack(mean_representations[layer])
    mean_representations[layer] = mean_representations[layer][ordering, :]

    output_path_layer = output_path.replace('.pt', f'_layer_{layer}.pt')
    torch.save(mean_representations[layer], output_path_layer)































