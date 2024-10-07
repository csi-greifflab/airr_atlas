"""
This script processes amino acid sequences from a FASTA file and computes their mean representations using a pre-trained ESM-2 model. The mean representations can be pooled or unpooled and saved to an output file. Optionally, the script can also compute embeddings for specific CDR3 sequences within the amino acid sequences.

Usage:
    python esm2_cdr3.py --fasta_path <path_to_fasta_file> --output_path <path_to_output_file> [--cdr3_path <path_to_cdr3_csv>] [--context <context_length>] [--layers <layers>] [--pooling <True/False>]

Arguments:
    --fasta_path (str): Path to the input FASTA file containing amino acid sequences.
    --output_path (str): Path to the output file where embeddings will be saved. Multiple files will be created if multiple layers are specified with '--layers'.
    --cdr3_path (str, optional): Path to the CSV file containing CDR3 sequences. Required if calculating CDR3 sequence embeddings.
    --context (int, optional): Number of amino acids to include before and after the CDR3 sequence. Default is 0.
    --layers (str, optional): Representation layers to extract from the model. Default is the last layer. Example: '--layers -1 6' will output the last layer and the sixth layer.
    --pooling (bool, optional): Whether to pool the embeddings or not. Default is True.

Example:
    python esm2_cdr3.py --fasta_path sequences.fasta --output_path embeddings.pt --cdr3_path cdr3.csv --context 5 --layers -1 6 --pooling True

Dependencies:
    - os
    - csv
    - argparse
    - Bio (Biopython)
    - torch
    - esm (Facebook Research)
    - torch.utils.data (PyTorch)
"""
import os
import argparse
import csv
import torch
from Bio import SeqIO
from esm import FastaBatchedDataset, pretrained

# Constants
MODEL_NAME = "esm2_t33_650M_UR50D"
BATCH_SIZE = 8192 # works with Nvidia V100-32GB GPU

# Parsing command-line arguments for input and output file paths
def parse_arguments():
    """Parse command-line arguments for input and output file paths."""
    PARSER = argparse.ArgumentParser(description="Input path")
    PARSER.add_argument("--fasta_path", type=str, required=True,
                        help="Fasta path + filename.fa")
    PARSER.add_argument("--output_path", type=str, required=True,
                        help="Output path + filename.pt \nWill output multiple files if multiple layers are specified with '--layers'. Output file is a single tensor or a list of tensors when --pooling is False.")
    PARSER.add_argument("--cdr3_path", default = None, type=str,
                        help="Path to the CDR3 CSV file. Only required when calculating CDR3 sequence embeddings.")
    PARSER.add_argument("--context", default = 0,type=int,
                        help="Number of amino acids to include before and after CDR3 sequence")
    PARSER.add_argument("--layers", type=str, nargs='*', default="-1",
                        help="Representation layers to extract from the model. Default is the last layer. Example: argument '--layers -1 6' will output the last layer and the sixth layer.")
    PARSER.add_argument('--pooling', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to pool the embeddings or not. Default is True.")
    return PARSER.parse_args()



#debug values
#os.environ["PYTORCH_CUDA_ALLOC_CONF"]='expandable_segments:True'
#FASTA_FILE = '/doctorai/userdata/airr_atlas/data/sequences/test_500.fa'
#OUTPUT_FILE = '/doctorai/userdata/airr_atlas/embedding/test/test_cdr3.pt'
#LAYERS = list(range(1,33 + 1))
#LAYERS = list(range(1,33 + 1))
#CDR3_PATH = '/doctorai/userdata/airr_atlas/data/sequences/trastuzumab/trastuzumab_cdr3_heavy.csv'
#CONTEXT = None
#POOLING = False

def fasta_to_dict(fasta_file):
    """Convert FASTA file into a dictionary."""
    print('Loading and batching input sequences...')
    seq_dict = {}
    with open(fasta_file) as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq_dict[record.id] = " ".join(str(record.seq)) # AA tokens for hugging face models must be space gapped
            # print progress
            if len(seq_dict) % 1000 == 0:
                print(f'{len(seq_dict)} sequences loaded')
    return seq_dict

def load_cdr3(cdr3_path):
    """Load CDR3 sequences and store in a dictionary."""
    if cdr3_path:
        with open(cdr3_path) as f:
            reader = csv.reader(f)
            cdr3_dict = {rows[0]:rows[1] for rows in reader}
        return cdr3_dict
    else:
        return None

def initialize_model(model_name=MODEL_NAME):
    # Loading the pretrained model and alphabet for tokenization
    print("Loading model...")
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()  # Setting the model to evaluation mode

    # Moving the model to GPU if available for faster processing
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
        device = "cuda"
    return model, alphabet

def load_data(fasta_path, alphabet, batch_size=BATCH_SIZE):
    print('Loading and batching input sequences...')
    # Creating a dataset from the input fasta file
    dataset = FastaBatchedDataset.from_file(fasta_path)
    # Generating batch indices based on token count
    batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
    # DataLoader to iterate through batches efficiently
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {fasta_path} with {len(dataset)} sequences")
    return data_loader, batches

def load_layers(model, layers):
    # Checking if the specified representation layers are valid
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in layers)
    layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in layers]
    return layers

def compute_embeddings(data_loader, batches, model, layers, sequences, context, pooling, cdr3_path, cdr3_dict):
    """Compute embeddings for the sequences."""
    # Initializing lists to store mean representations and sequence labels
    mean_representations = {layer: [] for layer in layers}
    sequence_labels = []
    # Processing each batch without computing gradients (to save memory and computation)
    with torch.no_grad():
        total_batches = len(data_loader)
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )

            # Moving tokens to GPU if available
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            # Computing representations for the specified layers
            out = model(toks, repr_layers=layers, return_contacts=False)

            # Extracting layer representations and moving them to CPU
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            if cdr3_path is None:
                for counter, label in enumerate(labels):
                    sequence_labels.append(label)
                    for layer in layers:
                        if pooling:
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
                        cdr3_sequence = cdr3_dict[label]
                    except KeyError:
                        print(f'No cdr3 sequence found for {label}')
                        continue
                    
                    #print(f'Processing {label}')
                    full_sequence = sequences[label]

                    # remove '-' from cdr3_sequence
                    cdr3_sequence = cdr3_sequence.replace('-', '')

                    # get position of cdr3_sequence in sequence
                    start = max(full_sequence.find(cdr3_sequence) - context, 0)
                    end = max(start + len(cdr3_sequence) + context, len(full_sequence))
                    sequence_labels.append(label)
                    for layer in layers:
                        if pooling:
                            mean_representation = representations[layer][i, start : end].mean(0).clone()
                        else:
                            mean_representation = representations[layer][i, start : end].clone()
                        # We take mean_representation[0] to keep the [array] instead of [[array]].
                        mean_representations[layer].append(mean_representation)
            # print the progress in one line
            print(f"Batch {batch_idx + 1}/{total_batches} completed", end='\r')
    print('Finished processing sequences')
    # Clear GPU memory
    print("Clearing GPU memory...")
    torch.cuda.empty_cache()
    return mean_representations, sequence_labels

def export_embeddings(mean_representations, layers, output_path, context=None, pooling=True):
    """Stack representations of each layer into a single tensor and save to output file."""
    for layer in layers:
        if context:
            output_file_layer = output_path.replace('.pt', f'_context_{context}_layer_{layer}.pt')
        else:
            output_file_layer = output_path.replace('.pt', f'_layer_{layer}.pt')
        if pooling:
            mean_representations[layer] = torch.vstack(mean_representations[layer])
        else:
            output_file_layer = output_file_layer.replace('.pt', '_full.pt')
        torch.save(mean_representations[layer], output_file_layer)
        print(f"Saved mean representations for layer {layer} to {output_file_layer}")

def export_sequence_indices(sequence_labels, output_path):
    """Save sequence indices to a CSV file."""
    output_file_idx = output_path.replace('.pt', '_idx.csv')
    with open(output_file_idx, 'w') as f:
        f.write('index,sequence_id\n')
        for i, label in enumerate(sequence_labels):
            f.write(f'{i},{label}\n')
    print(f"Saved sequence indices to {output_file_idx}")


def main():
    args = parse_arguments()
    # Store arguments
    fasta_path = args.fasta_path
    output_path = args.output_path
    cdr3_path = args.cdr3_path
    context = args.context
    layers = list(map(int, args.layers[0].split()))
    pooling = bool(args.pooling)

    # Read sequences from the FASTA file
    sequences = fasta_to_dict(fasta_path)

    # Check if output directory exists and creates it if it's missing
    if not os.path.exists(os.path.dirname(output_path)):
        # if the directory is not present create it.
        os.makedirs(os.path.dirname(output_path))

    # load CDR3 sequences if given
    cdr3_dict = load_cdr3(cdr3_path)

    # Initialize model and prepare input data
    model, alphabet = initialize_model(MODEL_NAME)
    layers = load_layers(model, layers)
    data_loader, batches = load_data(fasta_path, alphabet, BATCH_SIZE)

    # Compute embeddings
    mean_representations, sequence_labels = compute_embeddings(
        data_loader, batches, model, layers, sequences,
        context, pooling, cdr3_path, cdr3_dict
        )

    # Write embeddings to disk and export sequence indices
    export_embeddings(mean_representations, layers, output_path, context, pooling)
    export_sequence_indices(sequence_labels, output_path)

if __name__ == "__main__":
    main()
