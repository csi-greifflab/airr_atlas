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

import sys

sys.path.append(
    "/doctorai/userdata/airr_atlas/"
)  # Add the project directory to the Python path
import os
import torch
from esm import FastaBatchedDataset, pretrained
from scripts.embedding_utils import (
    parse_arguments,
    fasta_to_dict,
    load_cdr3,
    export_embeddings,
    export_sequence_indices,
)
import time

# Constants
MODEL_NAME = "esm2_t33_650M_UR50D"
BATCH_SIZE = 30000  # works with Nvidia V100-32GB GPU


def initialize_model(model_name=MODEL_NAME):
    """Initialize the model, tokenizer"""
    #  Loading the pretrained model and alphabet for tokenization
    print("Loading model...")
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()  # Setting the model to evaluation mode

    # Moving the model to GPU if available for faster processing
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    return model, alphabet


def load_data(fasta_path, alphabet, batch_size=BATCH_SIZE):
    print("Loading and batching input sequences...")
    # Creating a dataset from the input fasta file
    dataset = FastaBatchedDataset.from_file(fasta_path)
    # Generating batch indices based on token count
    batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
    # DataLoader to iterate through batches efficiently
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {fasta_path} with {len(dataset)} sequences")
    return data_loader


def load_layers(model, layers):
    # Checking if the specified representation layers are valid
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in layers)
    layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in layers]
    return layers


def compute_embeddings(
    data_loader, model, layers, sequences, context, pooling, cdr3_path, cdr3_dict
):
    """Compute embeddings for the sequences."""
    # Initializing lists to store mean representations and sequence labels
    mean_representations = {layer: [] for layer in layers}
    sequence_labels = []
    # Processing each batch without computing gradients (to save memory and computation)
    with torch.no_grad():
        total_batches = len(data_loader)
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            start_time = time.time()
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
                            mean_representation = (
                                representations[layer][
                                    counter, 1 : len(strs[counter]) + 1
                                ]
                                .mean(0)
                                .clone()
                            )
                        else:
                            mean_representation = representations[layer][
                                counter, 1 : len(strs[counter]) + 1
                            ].clone()
                        # We take mean_representation[0] to keep the [array] instead of [[array]].
                        mean_representations[layer].append(mean_representation)
            else:
                # Mean pooling representations for each sequence,
                # excluding the beginning-of-sequence (bos) token
                for i, label in enumerate(labels):
                    try:
                        cdr3_sequence = cdr3_dict[label]
                    except KeyError:
                        print(f"No cdr3 sequence found for {label}")
                        continue

                    # print(f'Processing {label}')
                    full_sequence = sequences[label]

                    # remove '-' from cdr3_sequence
                    cdr3_sequence = cdr3_sequence.replace("-", "")

                    # get position of cdr3_sequence in sequence
                    start = max(full_sequence.find(cdr3_sequence) - context, 0)
                    end = min(start + len(cdr3_sequence) + context, len(full_sequence))
                    sequence_labels.append(label)
                    for layer in layers:
                        if pooling:
                            mean_representation = (
                                representations[layer][i, start:end].mean(0).clone()
                            )
                        else:
                            mean_representation = representations[layer][
                                i, start:end
                            ].clone()
                        # We take mean_representation[0] to keep the [array] instead of [[array]].
                        mean_representations[layer].append(mean_representation)

            # Calculate tokens per second
            tokens_per_sec = round(BATCH_SIZE / (time.time() - start_time), 2)
            # print the progress in one line
            print(
                f"ESM2:       Batch {batch_idx + 1}/{total_batches} completed. {tokens_per_sec} toks/s",
                end="\r",
            )
    print("Finished processing sequences")
    # Clear GPU memory
    print("Clearing GPU memory...")
    torch.cuda.empty_cache()
    return mean_representations, sequence_labels


def main():
    # Parse and store arguments
    args = parse_arguments()
    fasta_path = args.fasta_path
    output_path = args.output_path
    cdr3_path = args.cdr3_path
    context = args.context
    layers = list(map(int, args.layers.strip().split()))
    pooling = bool(args.pooling)

    # Print summary of arguments
    print(f"FASTA file: {fasta_path}")
    print(f"Output file: {output_path}")
    print(f"CDR3 file: {cdr3_path}")
    print(f"Context: {context}")
    print(f"Layers: {layers}")
    print(f"Pooling: {pooling}\n")

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
    data_loader = load_data(fasta_path, alphabet, BATCH_SIZE)

    # Compute embeddings
    mean_representations, sequence_labels = compute_embeddings(
        data_loader, model, layers, sequences, context, pooling, cdr3_path, cdr3_dict
    )

    # Write embeddings to disk and export sequence indices
    export_embeddings(mean_representations, layers, output_path, context, pooling)
    export_sequence_indices(sequence_labels, output_path)


if __name__ == "__main__":
    main()
# sys.argv = ['esm2_cdr3.py',
#            '--fasta_path','/doctorai/userdata/airr_atlas/data/sequences/test.fa',
#            '--cdr3_path', '/doctorai/userdata/airr_atlas/data/sequences/trastuzumab/tz_cdr3.csv',
#            '--output_path','/doctorai/userdata/airr_atlas/data/embeddings/test.pt',
#            '--layers', "1"]
