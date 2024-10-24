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

# Parsing command-line arguments for input and output file paths
parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("fasta_path", type=str, help="Fasta path + filename.fa")
parser.add_argument("output_path", type=str, help="Output path + filename.pt")
args = parser.parse_args()

# Storing the input and output file paths
fasta_file = args.fasta_path
output_file = args.output_path
output_file_bos = args.output_path.replace(".pt", "_bos.pt")

# Pre-defined model location and batch token size
MODEL_LOCATION = "esm2_t33_650M_UR50D"
TOKS_PER_BATCH = 4096
REPR_LAYERS = [-1]  # [-1] is a default value to extract the last layer

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
mean_representations = []
bos_representations = []
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
            seq_labels.append(label)
            mean_representation = [t[i, 1 : len(strs[i]) + 1].mean(0).clone()
                    for layer, t in representations.items()]
            # We take mean_representation[0] to keep the [array] instead of [[array]].
            mean_representations.append(mean_representation[0])
            bos_representation = [t[i, 0].clone() 
                                  for layer, t in representations.items()]
            bos_representations.append(bos_representation[0])
            
# Stacking all mean representations into a single tensor
mean_representations = torch.vstack(mean_representations)
bos_representations = torch.vstack(bos_representations)

# Sorting the representations based on sequence labels
ordering = np.argsort([int(i) for i in seq_labels])
mean_representations = mean_representations[ordering, :]
bos_representations = bos_representations[ordering, :]

# Saving the tensor to the specified output file
torch.save(mean_representations, output_file)
torch.save(bos_representations, output_file_bos)
