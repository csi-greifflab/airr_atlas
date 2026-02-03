import argparse
import os

import pandas as pd
import numpy as np
import edlib
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

import sys


def compute_pairwise_distance_in_memory(strings, n_jobs, max_dist=None):
    """
    Compute the full pairwise Levenshtein distance matrix in memory using Edlib,
    parallelizing by row.

    Args:
        strings (List[str]): list of sequences
        n_jobs (int): number of parallel jobs
        max_dist (int | None): if set, Edlib will early-exit when distance > max_dist

    Returns:
        np.ndarray[int16]: symmetric distance matrix
    """
    n = len(strings)
    # allocate int16 to avoid overflow for distances >127
    mat = np.zeros((n, n), dtype=np.int8)

    def compute_row(i):
        row = np.zeros(n, dtype=np.int8)
        s_i = strings[i]
        for j in range(i, n):
            result = edlib.align(s_i, strings[j],
                                 mode='NW',
                                 task='distance',
                                 k=max_dist if max_dist is not None else -1)
            row[j] = result['editDistance']
        return i, row

    # parallel compute each row of the upper triangle
    
    with tqdm_joblib(desc="Computing pairwise distances", total=n) as progress_bar:
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_row)(i) for i in range(n)
        )
    # fill matrix
    for i, row in results:
        mat[i, i:] = row[i:]
        mat[i:, i] = row[i:]
    return mat


def compute_pairwise_distance_memmap(strings, n_jobs, batch_size, filename, max_dist=None):
    """
    Compute the full pairwise Levenshtein distance matrix on disk using np.memmap + Edlib,
    writing in row‐blocks to minimize I/O overhead.

    Args:
        strings (List[str]): list of sequences
        n_jobs (int): number of parallel jobs
        batch_size (int): number of rows per block write
        filename (str): path for the memmap file (will be created/truncated)
        max_dist (int | None): if set, Edlib will early-exit when distance > max_dist

    Returns:
        np.memmap: memmap of the symmetric distance matrix
    """
    n = len(strings)
    # create or overwrite file
    mat = np.memmap(filename, dtype=np.int8, mode='w+', shape=(n, n))

    def process_block(start_row):
        end_row = min(n, start_row + batch_size)
        block_rows = end_row - start_row
        block = np.zeros((block_rows, n), dtype=np.int8)
        for bi in range(block_rows):
            i = start_row + bi
            s_i = strings[i]
            for j in range(n):
                result = edlib.align(s_i, strings[j],
                                     mode='NW',
                                     task='distance',
                                     k=max_dist if max_dist is not None else -1)
                block[bi, j] = result['editDistance']
        # write entire block at once
        mat[start_row:end_row, :] = block

    # parallel blocks
    starts = list(range(0, n, batch_size))
    Parallel(n_jobs=n_jobs)(
        delayed(process_block)(sr) for sr in starts
    )

    # fill lower triangle
    tril_i, tril_j = np.tril_indices(n, -1)
    mat[tril_i, tril_j] = mat[tril_j, tril_i]
    mat.flush()
    return mat



sys.argv = [
'get_LD_matrix1.py',  # Keep the script name as first element
'--input_metadata', "/doctorai/userdata/airr_atlas/data/sequences/bcr/antigens/covabdab_AND_background.tsv",
'--sequence_column', 'VH_VL',
'--output_file', '/doctorai/niccoloc/LD_covabdab_bg2.npy',
'--batch_size', '500',
'--n_jobs', '30',
# '--memmap',
'--max_dist', '10'
]

sys.argv = [
    'get_LD_matrix1.py',  # Keep the script name as first element
'--input_metadata', "/doctorai/userdata/airr_atlas/data/sequences/bcr/antigens/alphaseq_sars/metadata_alphaseq_HB_LB.csv",
'--sequence_column', 'Sequence',
'--output_file', '/doctorai/niccoloc/LD_alphaseq_HB_LB.npy',
'--batch_size', '500',
'--n_jobs', '30',
# '--memmap',
'--max_dist', '10'
]

 




def parse_arguments():
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--input_metadata', type=str, required=True,
                        help='Path to the input metadata file (CSV or TSV).')
    parser.add_argument('--sequence_column', type=str, required=True,
                        help='Name of the column containing sequences.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output file (Numpy array).')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for memmap processing.')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs.')
    parser.add_argument('--memmap', action='store_true',
                        help='Use memmap for large datasets.')
    parser.add_argument('--max_dist', type=int, default=None,
                        help='Maximum distance for early-exit in Edlib.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output.')
    return parser.parse_args()

args= parse_arguments()

print(f"Loading data from {args.input_metadata}...")
# allow auto-detect of separator (csv or tsv)
data = pd.read_csv(args.input_metadata, sep=None, engine='python')
print(f" Header of the data: {data.head()}")
if args.sequence_column not in data.columns:
    raise ValueError(f"Column '{args.sequence_column}' not found in input file.")

strings = data[args.sequence_column].astype(str).tolist()
print(f"Found {len(strings)} sequences.")

if args.memmap:
    # ensure output has .dat extension for memmap
    out = args.output_file
    if not out.endswith('.dat'):
        base, _ = os.path.splitext(out)
        out = base + '.dat'
    mat = compute_pairwise_distance_memmap(
        strings=strings,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        filename=out,
        max_dist=args.max_dist
    )
    print(f"Memmap distance matrix written to {out}")
else:
    mat = compute_pairwise_distance_in_memory(
        strings=strings,
        n_jobs=args.n_jobs,
        max_dist=args.max_dist
    )
    # save as .npy
    np.save(args.output_file, mat)
    print(f"In-memory distance matrix saved to {args.output_file}")
