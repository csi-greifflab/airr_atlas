import argparse
from Bio import SeqIO
import csv
import torch


# Parsing command-line arguments for input and output file paths
def parse_arguments():
    """Parse command-line arguments for input and output file paths."""
    PARSER = argparse.ArgumentParser(description="Input path")
    PARSER.add_argument(
        "--fasta_path", type=str, required=True, help="Fasta path + filename.fa"
    )
    PARSER.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path + filename.pt \nWill output multiple files if multiple layers are specified with '--layers'. Output file is a single tensor or a list of tensors when --pooling is False.",
    )
    PARSER.add_argument(
        "--cdr3_path",
        default=None,
        type=str,
        help="Path to the CDR3 CSV file. Only required when calculating CDR3 sequence embeddings.",
    )
    PARSER.add_argument(
        "--context",
        default=0,
        type=int,
        help="Number of amino acids to include before and after CDR3 sequence",
    )
    PARSER.add_argument(
        "--layers",
        type=str,
        nargs="?",
        default=["-1"],
        help="Representation layers to extract from the model. Default is the last layer. Example: argument '--layers -1 6' will output the last layer and the sixth layer.",
    )
    PARSER.add_argument(
        "--pooling",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Whether to pool the embeddings or not. Default is True.",
    )
    return PARSER.parse_args()


def fasta_to_dict(fasta_file, gaps=False):
    """Convert FASTA file into a dictionary."""
    print("Loading and batching input sequences...")
    seq_dict = {}
    with open(fasta_file) as f:
        for record in SeqIO.parse(f, "fasta"):
            if gaps:
                seq_dict[record.id] = " ".join(
                    str(record.seq)
                )  # AA tokens for hugging face models must be space gapped
            else:
                seq_dict[record.id] = str(record.seq)
            # print progress
            if len(seq_dict) % 1000 == 0:
                print(f"{len(seq_dict)} sequences loaded", end="\r")
    return seq_dict


def load_cdr3(cdr3_path):
    """Load CDR3 sequences and store in a dictionary."""
    if cdr3_path:
        with open(cdr3_path) as f:
            reader = csv.reader(f)
            cdr3_dict = {rows[0]: rows[1] for rows in reader}
        return cdr3_dict
    else:
        return None


def export_embeddings(
    mean_representations, layers, output_path, context=None, pooling=True
):
    """Stack representations of each layer into a single tensor and save to output file."""
    for layer in layers:
        if context:
            output_file_layer = output_path.replace(
                ".pt", f"_context_{context}_layer_{layer}.pt"
            )
        else:
            output_file_layer = output_path.replace(".pt", f"_layer_{layer}.pt")
        if pooling:
            mean_representations[layer] = torch.vstack(mean_representations[layer])
        else:
            output_file_layer = output_file_layer.replace(".pt", "_unpooled.pt")
        torch.save(mean_representations[layer], output_file_layer)
        print(f"Saved mean representations for layer {layer} to {output_file_layer}")


def export_sequence_indices(sequence_labels, output_path):
    """Save sequence indices to a CSV file."""
    output_file_idx = output_path.replace(".pt", "_idx.csv")
    with open(output_file_idx, "w") as f:
        f.write("index,sequence_id\n")
        for i, label in enumerate(sequence_labels):
            f.write(f"{i},{label}\n")
    print(f"Saved sequence indices to {output_file_idx}")
