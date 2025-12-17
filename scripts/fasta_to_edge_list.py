import os
import time
import argparse
from collections import Counter
import atriegc
from Levenshtein import distance as lev_distance
from pybktree import BKTree
from rapidfuzz.distance.Levenshtein import distance as rapidfuzz_lev_distance
from alive_progress import alive_bar

parser = argparse.ArgumentParser(description="Convert a FASTA file to an edge list.")
parser.add_argument(
    "fasta_file",
    type=str,
    help="Path to the input FASTA file containing sequences.",
)
parser.add_argument(
    "output_file",
    type=str,
    help="Path to the output file where the edge list will be saved.",
)
parser.add_argument(
    "--distance_metric",
    type=str,
    choices=["hamming", "levenshtein"],
    help="Distance metric to use for edge creation.",
    required=True,
)
parser.add_argument(
    "--max_distance",
    type=int,
    default=1,
    nargs="?",
    help="Maximum distance between sequences to be considered as edges.",
)
parser.add_argument(
    "--keep_first_duplicate",
    action="store_true",
    help="Keep the first duplicate sequence and remove the rest.",
)
parser.add_argument(
    "--keep_self_edges",
    action="store_true",
    help="Keep self edges in the output.",
)
parser.add_argument(
    "--weighted",
    action="store_true",
    help="Output a weighted edge list.",
)


def read_fasta(fasta_file):
    sequences = {}
    current_id = None
    current_seq = []
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line.strip()[1:]
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_id:
            sequences[current_id] = "".join(current_seq)
    return sequences


def add_edge(edge_set, src, tgt, weight=None):
    if weight is not None:
        edge_set.add((src, tgt, weight))
    else:
        edge_set.add((src, tgt))


def normalize_edges(edge_set):
    return set((min(a, b), max(a, b)) for a, b, *_ in edge_set)


def hamming_dist(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def hamming(sequences, keep_self_edges=False, max_distance=1, weighted=False):
    # Insert sequences into the atriegc database
    tr = atriegc.TrieAA()
    for sequence in sequences.values():
        tr.insert(sequence)

    edge_set = set()
    len_sequences = len(sequences)
    with alive_bar(len_sequences) as bar:
        for sequence_id, sequence in sequences.items():
            neighbors = tr.neighbours(sequence, max_distance)
            for neighbor in neighbors:
                if neighbor in seq_to_key:
                    neighbor_id = seq_to_key[neighbor]
                    if keep_self_edges:
                        if weighted:
                            weight = hamming_dist(sequence, neighbor)
                            add_edge(edge_set, sequence_id, neighbor_id, weight)
                        else:
                            add_edge(edge_set, sequence_id, neighbor_id)
                    elif sequence_id != neighbor_id:  # avoid self-loop
                        if weighted:
                            weight = hamming(sequence, neighbor)
                            add_edge(edge_set, sequence_id, neighbor_id, weight)
                        else:
                            add_edge(edge_set, sequence_id, neighbor_id)
            bar()
    edge_set = normalize_edges(edge_set)
    return edge_set


def levenshtein(
    sequences,
    keep_self_edges=False,
    dist_fun=lev_distance,
    max_distance=1,
    weighted=False,
):
    bk = BKTree(dist_fun)
    for seq in sequences.values():
        bk.add(seq)
    edge_set = set()
    len_sequences = len(sequences)
    with alive_bar(len_sequences) as bar:
        for sequence_id, sequence in sequences.items():
            neighbors = bk.find(sequence, max_distance)
            for neighbor in neighbors:
                distance = neighbor[0]
                neighbor_id = seq_to_key[neighbor[1]]
                if keep_self_edges:
                    if weighted:
                        add_edge(edge_set, sequence_id, neighbor_id, distance)
                    else:
                        add_edge(edge_set, sequence_id, neighbor_id)
                elif sequence_id != neighbor_id:  # avoid self-loop
                    if weighted:
                        add_edge(edge_set, sequence_id, neighbor_id, distance)
                    else:
                        add_edge(edge_set, sequence_id, neighbor_id)
            bar()
    edge_set = normalize_edges(edge_set)
    return edge_set


if __name__ == "__main__":
    args = parser.parse_args()

    fasta_file = args.fasta_file
    # fasta_file = "data/sequences/bcr/trastuzumab/tz_heavy_chain_100k.fa"
    output_file = args.output_file
    # output_file = "edge_lists/tz_heavy_chain_100k_edges.csv"
    # max_distance = 1
    keep_self_edges = args.keep_self_edges
    max_dist = args.max_distance
    weighted = args.weighted
    if max_dist == 1:
        weighted = False
        print("Warning: max_distance is set to 1. The output will not be weighted.")

    # Read the sequences from the FASTA file

    sequences = read_fasta(fasta_file)
    # Deduplicate sequences
    if args.keep_first_duplicate:
        seen = set()
        sequences = {
            k: v for k, v in sequences.items() if not (v in seen or seen.add(v))
        }
    else:
        # Count the occurrences of each sequence
        sequence_counts = Counter(sequences.values())
        # Keep only the first occurrence of each sequence
        sequences = {k: v for k, v in sequences.items() if sequence_counts[v] == 1}

    seq_to_key = {v: k for k, v in sequences.items()}

    if args.distance_metric == "levenshtein":
        print("Using Levenshtein distance metric.")
        edge_set = levenshtein(sequences, dist_fun=rapidfuzz_lev_distance)
    elif args.distance_metric == "hamming":
        print("Using Hamming distance metric.")
        edge_set = hamming(sequences, max_distance=max_dist)

    print(f"Found {len(edge_set)} edges. Outputting to {output_file}.")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
