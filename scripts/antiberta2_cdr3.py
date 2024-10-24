"""
This script processes amino acid sequences from a FASTA file and computes their embeddings using the RoFormer model from Hugging Face's Transformers library. The embeddings can be pooled or unpooled and saved to an output file. Optionally, the script can also compute embeddings for specific CDR3 sequences within the amino acid sequences.

Usage:
    python antiberta2_cdr3.py --fasta_path <path_to_fasta_file> --output_path <path_to_output_file> [--cdr3_path <path_to_cdr3_csv>] [--context <context_length>] [--layers <layers>] [--pooling <True/False>]

Arguments:
    --fasta_path (str): Path to the input FASTA file containing amino acid sequences.
    --output_path (str): Path to the output file where embeddings will be saved. Multiple files will be created if multiple layers are specified with '--layers'.
    --cdr3_path (str, optional): Path to the CSV file containing CDR3 sequences. Required if calculating CDR3 sequence embeddings.
    --context (int, optional): Number of amino acids to include before and after the CDR3 sequence. Default is 0.
    --layers (str, optional): Representation layers to extract from the model. Default is the last layer. Example: '--layers -1 6' will output the last layer and the sixth layer.
    --pooling (bool, optional): Whether to pool the embeddings or not. Default is True.

Example:
    python antiberta2_cdr3.py --fasta_path sequences.fasta --output_path embeddings.pt --cdr3_path cdr3.csv --context 5 --layers -1 6 --pooling True

Dependencies:
    - os
    - csv
    - argparse
    - Bio (Biopython)
    - torch
    - transformers (Hugging Face)
    - torch.utils.data (PyTorch)
"""

import sys

sys.path.append(
    "/doctorai/userdata/airr_atlas/"
)  # Add the project directory to the Python path
import os
import torch
import time
from transformers import RoFormerTokenizer, RoFormerModel
from torch.utils.data import DataLoader, TensorDataset
from scripts.embedding_utils import (
    parse_arguments,
    fasta_to_dict,
    load_cdr3,
    export_embeddings,
    export_sequence_indices,
)

# Constants
MODEL_NAME = "alchemab/antiberta2-cssp"
BATCH_SIZE = 512  # Adjust based on your GPU's memory
MAX_LENGTH = 200


def initialize_model(model_name):
    """Initialize the model, tokenizer, and device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RoFormerTokenizer.from_pretrained(model_name)
    model = RoFormerModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, device, tokenizer


def load_layers(model, layers):
    """Check if the specified representation layers are valid."""
    assert all(
        -(model.config.num_hidden_layers + 1) <= i <= model.config.num_hidden_layers
        for i in layers
    )
    layers = [
        (i + model.config.num_hidden_layers + 1) % (model.config.num_hidden_layers + 1)
        for i in layers
    ]
    return layers


def load_data(sequences, tokenizer, batch_size, max_length=MAX_LENGTH):
    """Tokenize sequences and create a DataLoader."""
    # Tokenize sequences
    input_ids = []
    attention_masks = []
    total_sequences = len(sequences)
    print("Start tokenization")
    for counter, sequence in enumerate(sequences.values()):
        # tokens = tokenizer(sequence,truncation=True, padding='max_length', return_tensors="pt",add_special_tokens=True, max_length=200)
        # tokenize sequences without truncation
        tokens = tokenizer(
            sequence,
            truncation=False,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
            max_length=max_length,
        )
        # print( tokenizer.decode(tokens['input_ids'][0]))
        input_ids.append(tokens["input_ids"])
        attention_masks.append(tokens["attention_mask"])
        # Calculate and print the percentage of completion
        percent_complete = ((counter + 1) / total_sequences) * 100
        # Check and print the progress at each 2% interval
        if (counter + 1) == total_sequences or int(percent_complete) % 2 == 0:
            # Ensures the message is printed once per interval and at 100% completion
            if (counter + 1) == total_sequences or (
                int(percent_complete / 2)
                != int(((counter) / total_sequences) * 100 / 2)
            ):
                print(f"Progress: {percent_complete:.2f}%", end="\r")

    # Convert lists to tensors and create a dataset
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    dataset = TensorDataset(input_ids, attention_masks)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader


def compute_embeddings(
    data_loader,
    model,
    device,
    layers,
    sequences,
    context,
    pooling,
    cdr3_path,
    cdr3_dict,
    batch_size,
):
    """Compute embeddings for the sequences."""
    # Initializing lists to store mean representations and sequence labels
    mean_representations = {layer: [] for layer in layers}
    sequence_labels = []
    # Processing each batch without computing gradients (to save memory and computation)
    with torch.no_grad():
        total_batches = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            start_time = time.time()
            labels = list(sequences.keys())[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            input_ids, attention_mask = [b.to(device, non_blocking=True) for b in batch]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Extracting layer representations and moving them to CPU
            representations = {
                layer: outputs.hidden_states[layer].to(device="cpu") for layer in layers
            }
            # TODO add optional argument to return mean pooled full embedding even if cdr3_path is specified
            if cdr3_path is None:
                # Append labels to SEQUENCE_LABELS
                sequence_labels.extend(labels)
                for layer in layers:
                    if pooling:
                        mean_representations[layer].extend(
                            representations[layer][
                                i, 1 : len(sequences[label].replace(" ", "")) + 1
                            ]
                            .mean(0)
                            .clone()
                            for i, label in enumerate(labels)
                        )
                    else:
                        mean_representations[layer].extend(
                            representations[layer][
                                i, 1 : len(sequences[label].replace(" ", "")) + 1
                            ].clone()
                            for i, label in enumerate(labels)
                        )
            else:
                for counter, label in enumerate(labels):
                    try:
                        cdr3_sequence = cdr3_dict[label]
                    except KeyError:
                        print(f"No cdr3 sequence found for {label}")
                        continue
                    # print(f'Processing {label}')

                    # load sequence without spaces
                    full_sequence = sequences[label].replace(" ", "")

                    # remove '-' from cdr3_sequence
                    cdr3_sequence = cdr3_sequence.replace("-", "")

                    # get position of cdr3_sequence in sequence
                    start = max(full_sequence.find(cdr3_sequence) - context, 0)
                    end = min(start + len(cdr3_sequence) + context, len(full_sequence))
                    sequence_labels.append(label)

                    for layer in layers:
                        if pooling:
                            mean_representation = (
                                representations[layer][counter, start:end]
                                .mean(0)
                                .clone()
                            )
                        else:
                            mean_representation = representations[layer][
                                counter, start:end
                            ].clone()
                        # We take mean_representation[0] to keep the [array] instead of [[array]].
                        mean_representations[layer].append(mean_representation)

            # Calculate tokens per second
            tokens_per_sec = round(BATCH_SIZE / (time.time() - start_time), 2)
            # print the progress in one line
            print(
                f"AntiBERTa2:     Batch {batch_idx + 1}/{total_batches} completed. {tokens_per_sec} toks/s",
                end="\r",
            )
    # Clear GPU memory
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
    sequences = fasta_to_dict(fasta_path, gaps=True)

    # Check if output directory exists and creates it if it's missing
    if not os.path.exists(os.path.dirname(output_path)):
        # if the directory is not present create it.
        os.makedirs(os.path.dirname(output_path))

    # load CDR3 sequences if given
    cdr3_dict = load_cdr3(cdr3_path)

    # Initialize model and prepare input data
    model, device, tokenizer = initialize_model(MODEL_NAME)
    layers = load_layers(model, layers)
    data_loader = load_data(sequences, tokenizer, BATCH_SIZE, max_length=MAX_LENGTH)

    # Compute embeddings
    mean_representations, sequence_labels = compute_embeddings(
        data_loader,
        model,
        device,
        layers,
        sequences,
        context,
        pooling,
        cdr3_path,
        cdr3_dict,
        BATCH_SIZE,
    )

    # Write embeddings to disk and export sequence indices
    export_embeddings(mean_representations, layers, output_path, context, pooling)
    export_sequence_indices(sequence_labels, output_path)


if __name__ == "__main__":
    main()
