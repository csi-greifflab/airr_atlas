# pylint: disable=import-error
"""
This script takes a fasta file as input and outputs a tensor of the mean representations of each sequence in the file using a pre-trained ESM-2 model.
The tensor is saved as a PyTorch file.

Args:
    fasta_path (str): Path to the fasta file.
    output_path (str): Path to save the output tensor.
"""
import os
import argparse
import csv
import torch
from Bio import SeqIO
from esm import FastaBatchedDataset, pretrained


# Parsing command-line arguments for input and output file paths
PARSER = argparse.ArgumentParser(description="Input path")
PARSER.add_argument("--fasta_path", type=str, required=True,
                    help="Fasta path + filename.fa")
PARSER.add_argument("--output_path", type=str, required=True,
                    help="Output path + filename.pt \nWill output multiple files if multiple layers are specified with '--layers'.")
PARSER.add_argument("--cdr3_path", default = None, type=str,
                    help="Path to the CDR3 CSV file. Only required when calculating CDR3 sequence embeddings.")
PARSER.add_argument("--context", default = 0,type=int,
                    help="Number of amino acids to include before and after CDR3 sequence")
PARSER.add_argument("--layers", type=str, nargs='*', default="-1",
                    help="Representation layers to extract from the model. Default is the last layer. Example: argument '--layers -1 6' will output the last layer and the sixth layer.")
PARSER.add_argument('--pooling', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to pool the embeddings or not. Default is True.")
ARGS = PARSER.parse_args()

# Storing the input arguments in variables
FASTA_FILE = ARGS.fasta_path
OUTPUT_FILE = ARGS.output_path
CDR3_PATH = ARGS.cdr3_path
CONTEXT = ARGS.context
LAYERS = list(map(int, ARGS.layers[0].split()))
if ARGS.pooling:
    POOLING = ARGS.pooling
else:
    POOLING = False

# Print summary of arguments
print(f"FASTA file: {FASTA_FILE}")
print(f"Output file: {OUTPUT_FILE}")
if CDR3_PATH:
    print(f"CDR3 file: {CDR3_PATH}")
if CONTEXT:
    print(f"Context: {CONTEXT}")
print(f"Layers: {LAYERS}")
print(f"Pooling: {POOLING}\n")





#debug values
os.environ["PYTORCH_CUDA_ALLOC_CONF"]='expandable_segments:True'
#FASTA_FILE = '/doctorai/userdata/airr_atlas/data/sequences/test_500.fa'
#OUTPUT_FILE = '/doctorai/userdata/airr_atlas/embedding/test/test_cdr3.pt'
#LAYERS = list(range(1,33 + 1))
#LAYERS = list(range(1,33 + 1))
#CDR3_PATH = '/doctorai/userdata/airr_atlas/data/sequences/trastuzumab/trastuzumab_cdr3_heavy.csv'
#CONTEXT = None
#POOLING = False

# Check if output directory exists and creates it if it's missing
if not os.path.exists(os.path.dirname(OUTPUT_FILE)):

    # if the demo_folder directory is not present  
    # then create it. 
    os.makedirs(os.path.dirname(OUTPUT_FILE))
# Load cdr3 sequences and store in dictionary
if CDR3_PATH:
    with open(CDR3_PATH) as f:
        reader = csv.reader(f)
        CDR3_DICT = {rows[0]:rows[1] for rows in reader}

# convert fasta into dictionary
def fasta_to_dict(fasta_file):
    """
    Converts a fasta file into a dictionary with sequence IDs as keys and sequences as values.

    Args:
        fasta_file (str): Path to the fasta file.

    Returns:
        dict: A dictionary with sequence IDs as keys and sequences as values.
    """
    print('Loading and batching input sequences...')
    seq_dict = {}
    with open(fasta_file) as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq_dict[record.id] = str(record.seq)
            # print progress
            if len(seq_dict) % 1000 == 0:
                print(f'{len(seq_dict)} sequences loaded')
    return seq_dict            


FASTA_SEQUENCES = fasta_to_dict(FASTA_FILE)

# make sure FASTA_SEQUENCES and CDR3_DICT have the same keys
if CDR3_PATH:
    FASTA_KEYS = set(FASTA_SEQUENCES.keys())
    CDR3_KEYS = set(CDR3_DICT.keys())
    missing_keys = FASTA_KEYS - CDR3_KEYS
    for key in missing_keys:
        FASTA_SEQUENCES.pop(key)

# TODO investigate missing_keys
if CDR3_PATH:
    missing_keys = [key for key in FASTA_SEQUENCES if key not in CDR3_DICT]

# Pre-defined model location and batch token size
MODEL_LOCATION = "esm2_t33_650M_UR50D"
TOKS_PER_BATCH = 50000 # works with Nvidia V100-32GB GPU

# Loading the pretrained model and alphabet for tokenization
print("Loading model...")
MODEL, ALPHABET = pretrained.load_model_and_alphabet(MODEL_LOCATION)
MODEL.eval()  # Setting the model to evaluation mode

# Moving the model to GPU if available for faster processing
if torch.cuda.is_available():
    MODEL = MODEL.cuda()
    print("Transferred model to GPU")

print('Loading and batching input sequences...')
# Creating a dataset from the input fasta file
DATASET = FastaBatchedDataset.from_file(FASTA_FILE)
# Generating batch indices based on token count
BATCHES = DATASET.get_batch_indices(TOKS_PER_BATCH, extra_toks_per_seq=1)
# DataLoader to iterate through batches efficiently
DATA_LOADER = torch.utils.data.DataLoader(
    DATASET, collate_fn=ALPHABET.get_batch_converter(), batch_sampler=BATCHES
)

print(f"Read {FASTA_FILE} with {len(DATASET)} sequences")

# Checking if the specified representation layers are valid
assert all(-(MODEL.num_layers + 1) <= i <= MODEL.num_layers for i in LAYERS)

LAYERS = [(i + MODEL.num_layers + 1) % (MODEL.num_layers + 1) for i in LAYERS]

# Initializing lists to store mean representations and sequence labels
mean_representations = {layer: [] for layer in LAYERS}
sequence_labels = []
# Processing each batch without computing gradients (to save memory and computation)
with torch.no_grad():
    for batch_idx, (labels, strs, toks) in enumerate(DATA_LOADER):
        print(
            f"Processing {batch_idx + 1} of {len(BATCHES)} batches ({toks.size(0)} sequences)"
        )
        

        # Moving tokens to GPU if available
        if torch.cuda.is_available():
            toks = toks.to(device="cuda", non_blocking=True)

        # Computing representations for the specified layers
        out = MODEL(toks, repr_layers=LAYERS, return_contacts=False)

        # Extracting layer representations and moving them to CPU
        representations = {
            layer: t.to(device="cpu") for layer, t in out["representations"].items()
        }

        if CDR3_PATH is None:
            for counter, label in enumerate(labels):
                sequence_labels.append(label)
                for layer in LAYERS:
                    if POOLING:
                        mean_representation = representations[layer][counter, 1: len(strs[counter]) + 1].mean(0).clone()
                    else:
                        mean_representation = representations[layer][counter, 1: len(strs[counter]) + 1].clone()
                    # We take mean_representation[0] to keep the [array] instead of [[array]].
                    mean_representations[layer].append(mean_representation)
        else:
            # Mean pooling representations for each sequence,
            # excluding the beginning-of-sequence (bos) token
            for i, label in enumerate(labels):
                try:
                    cdr3_sequence = CDR3_DICT[label]
                except KeyError:
                    if label not in missing_keys:
                        print(f'No cdr3 sequence found for {label}')
                    continue
                
                #print(f'Processing {label}')
                full_sequence = FASTA_SEQUENCES[label]



                # remove '-' from cdr3_sequence
                cdr3_sequence = cdr3_sequence.replace('-', '')

                # get position of cdr3_sequence in sequence
                start = full_sequence.find(cdr3_sequence) - CONTEXT
                end = start + len(cdr3_sequence) + CONTEXT
                sequence_labels.append(label)
                for layer in LAYERS:
                    if POOLING:
                        mean_representation = representations[layer][i, start : end].mean(0).clone()
                    else:
                        mean_representation = representations[layer][i, start : end].clone()
                    # We take mean_representation[0] to keep the [array] instead of [[array]].
                    mean_representations[layer].append(mean_representation)
print('Finished processing sequences')
# Clear GPU memory
print("Clearing GPU memory...")
torch.cuda.empty_cache()

print("Saving mean representations to output file...")
# Stacking all mean representations into a single tensor and save to output file

# Save mean pooled representations for each layer to a separate file
for layer in LAYERS:
    # concattenate sequence labels to the tensor
    mean_representations[layer] = torch.vstack(mean_representations[layer])

    output_file_layer = OUTPUT_FILE.replace('.pt', f'_layer_{layer}.pt')
    if not POOLING:
        output_file_layer = output_file_layer.replace('.pt', '_full.pt')
    torch.save(mean_representations[layer], output_file_layer)
    print(f"Saved mean representations for layer {layer} to {output_file_layer}")

# Save sequence labels to a csv file
OUTPUT_FILE_IDX = OUTPUT_FILE.replace('.pt', '_idx.csv')
with open(OUTPUT_FILE_IDX, 'w') as f:
    f.write('index,sequence_id\n')
    for i, label in enumerate(sequence_labels):
        f.write(f'{i},{label}\n')
print(f"Saved sequence indices to {OUTPUT_FILE_IDX}")