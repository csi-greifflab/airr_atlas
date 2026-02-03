import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#from umap import UMAP
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
import torch
# import umap.plot
import Levenshtein as lev
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import jensenshannon
import concurrent.futures
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import pickle
import sys
import os
from scipy.sparse import save_npz
sys.path.insert(0, '/doctorai/niccoloc/airr_atlas/scripts/Vicinity_code')

from Vicinity_analysis_class import Vicinity_analysis
from Vicinity_analysis_class import calculate_moran_index
from Vicinity_analysis_class import prepare_data_for_plotting
from Vicinity_analysis_class import prepare_data_for_plotting_LD_MAT
from Vicinity_analysis_class import run_ggplot_vicinity

from libpysal.weights import W

import argparse
from sklearn.preprocessing import scale
import tracemalloc

# TODO look for LD to total TZ dataset
# Simulate command-line arguments for debugging
# sys.argv = [
#     'Vicinity_pipeline.py',  # Script name (first argument in sys.argv)
#     '--analysis_name', 'VH_ab2_attMat_ALL_layer1_cosine_v7',
#     '--input_metadata', '/doctorai/niccoloc/tz_metadata_60k.csv',
#     '--input_embeddings', '/doctorai/niccoloc/attention_matrices_flat_avg_ALL_ab2.pt',
#     '--save_results',
#     '--radius_range', '0,6,0.2',
#     '--plot_results',
#     '--df_junction_colname', 'junction_aa',
#     '--df_affinity_colname', 'affinity',
#     '--sample_size', '0',
#     '--chosen_metric', 'cosine',
#     '--compute_LD'
# ]
# 
# 
# sys.argv = [
#     'Vicinity_pipeline.py' ,
#     '--analysis_name' ,'WHOLE_LD' ,
#     '--input_metadata' ,'/doctorai/niccoloc/trastuzumab_metadata.csv' ,
#     '--input_embeddings' ,'/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab/antiberta2/cdr3_only/100k_sample_trastuzmab_cdr3_heavy_only_antiberta2_layer_16.pt' ,
#     '--input_idx' , '/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis/antiberta2/100k_sample_trastuzmab_antiberta2_idx.csv',
#     '--compute_LD' ,
#     '--save_results'  ,
#     '--plot_results'  ,
#     '--sample_size', '4000' ,
#     '--LD_sample_size', '530000',
# ]
# 
# 
# sys.argv = [
#      'Vicinity_pipeline.py' ,
#     '--analysis_name' ,'debug_FAISS' ,
#      '--input_metadata' ,"/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv" ,
#      '--input_embeddings' ,"/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/esm2/embeddings_unpooled/tz_paired_chain_100k_esm2_esm2_embeddings_unpooled_layer_1.npy",
#      '--input_idx' , "/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/tz_paired_chain_100k_esm2_idx.csv",
#      '--compute_LD' ,
#       '--chosen_metric', 'cosine',
#       '--LD_matrix' , '/doctorai/niccoloc/tz_LD_dist_mat_HB_LB.npy',
#     #   '--precomputed_LD' , "/doctorai/niccoloc/Vicinity_results_100k/WHOLE_LD/LD_WHOLE_hb_lb_530k.csv",
# #     '--save_results'  ,
# #     '--plot_results'  ,
#       '--result_dir', '/doctorai/niccoloc/Vicinity_results_100k_DEBUG',
#       '--df_junction_colname', 'cdr3_aa',
#       '--df_affinity_colname', 'binding_label',
#      '--sample_size', '3000' ,
#      '--LD_sample_size', '530000',
#      '--skip_knn'
#  ]

# input_idx: /doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/tz_paired_chain_100k_esm2_idx.csv
# input_metadata: /doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv
# input_embeddings: /doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/esm2/attention_matrices_average_layer

sys.argv = [
     'Vicinity_pipeline.py' ,
     '--analysis_name' ,'test_adj' ,
     '--input_metadata' ,"/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv" ,
    #  '--input_embeddings' ,"/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/esm2/attention_matrices_average_layers/tz_cdr3_only_100k_esm2_attention_matrices_average_layers_layer_3.pt",
     '--input_embeddings' ,"/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab_npy/esm2_t33_650M_UR50D/embeddings_unpooled/tz_cdr3_100k_esm2_t33_650M_UR50D_embeddings_unpooled_layer_1.npy",
    #  '--input_embeddings' ,"/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab_full/esm2/embeddings/tz_cdr3_esm2_embeddings_layer_32.pt",
    #  '--input_idx' , "/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/tz_cdr3_only_100k_idx.csv",
     '--input_idx' , "/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab_npy/esm2_t33_650M_UR50D/tz_cdr3_100k_idx.csv",
    #  '--input_idx' , "/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab_full/tz_cdr3_idx.csv",

    #  '--compute_LD' ,
      '--chosen_metric', 'cosine',
      '--LD_matrix' , '/doctorai/niccoloc/tz_LD_dist_mat_HB_LB.npy',
    #   '--precomputed_LD' , "/doctorai/niccoloc/Vicinity_results_100k/WHOLE_LD/LD_WHOLE_hb_lb_530k.csv",
#     '--save_results'  ,
#     '--plot_results'  ,
      '--result_dir', '/doctorai/niccoloc/Vicinity_results_sample_test',
      '--df_junction_colname', 'cdr3_aa',
      '--df_affinity_colname', 'binding_label',
     '--sample_size', '0' ,
     '--LD_sample_size', '530000',
     '--skip_knn'
 ]
 


# sys.argv = [
#      'Vicinity_pipeline.py' ,
#      '--analysis_name' ,'test_adj' ,
#      '--input_idx' , "/doctorai/userdata/airr_atlas/data/embeddings/brian_hie/cr9114_hie_idx.csv",
#      '--input_embeddings' ,"/doctorai/userdata/airr_atlas/data/embeddings/brian_hie/antiberta2-cssp/embeddings/brian_hie_antiberta2-cssp_embeddings_layer_1.npy",
#      '--input_metadata' ,"/doctorai/userdata/airr_atlas/data/sequences/bcr/brian_hie/cr9114_hie_metadata.csv" ,


#       '--compute_LD' ,
#       '--chosen_metric', 'cosine',
#       '--LD_matrix' , '/doctorai/niccoloc/tz_LD_dist_mat_HB_LB.npy',
#     #   '--precomputed_LD' , "/doctorai/niccoloc/Vicinity_results_100k/WHOLE_LD/LD_WHOLE_hb_lb_530k.csv",
# #     '--save_results'  ,
# #     '--plot_results'  ,
#       '--result_dir', '/doctorai/niccoloc/Vicinity_results_sample_test',
#       '--df_junction_colname', 'Sequence',
#       '--df_affinity_colname', 'binding_label',
#      '--sample_size', '0' ,
#      '--LD_sample_size', '530000',
#      '--skip_knn'
#  ]

# sys.argv = [
#      'Vicinity_pipeline.py' ,
#      '--analysis_name' ,'test_adj_cov' ,
#      '--input_idx' , "/doctorai/userdata/airr_atlas/data/embeddings/covabdab/esm2_t33_650M_UR50D/covabdab_bg_ESM2_idx.csv",
#      '--input_embeddings' ,"/doctorai/userdata/airr_atlas/data/embeddings/covabdab/esm2_t33_650M_UR50D/embeddings/covabdab_esm2_t33_650M_UR50D_embeddings_layer_26.npy",
#      '--input_metadata' ,"/doctorai/userdata/airr_atlas/data/sequences/bcr/antigens/covabdab_AND_background.tsv" ,


#       '--compute_LD' ,
#       '--chosen_metric', 'cosine',
#       '--LD_matrix' , '/doctorai/niccoloc/tz_LD_dist_mat_HB_LB.npy',
#     #   '--precomputed_LD' , "/doctorai/niccoloc/Vicinity_results_100k/WHOLE_LD/LD_WHOLE_hb_lb_530k.csv",
# #     '--save_results'  ,
# #     '--plot_results'  ,
#       '--result_dir', '/doctorai/niccoloc/Vicinity_results_sample_test',
#       '--df_junction_colname', 'VH_VL',
#       '--df_affinity_colname', 'binding_label',
#      '--sample_size', '0' ,
#      '--LD_sample_size', '530000',
#      '--skip_knn'
#  ]
 





def parse_arguments():
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--analysis_name', type=str, required=True, help='Name of the analysis')
    parser.add_argument('--df_junction_colname', type=str, default='junction_aa', help='Column name for junction aa')
    parser.add_argument('--df_affinity_colname', type=str, default='affinity', help='Column name for affinity')
    parser.add_argument('--input_idx', type=str, default='', help='File containing the corrispondence between the embeddings index and the metadata id')
    parser.add_argument('--input_metadata', type=str, required=True, help='Path to input metadata CSV file')
    parser.add_argument('--input_embeddings', type=str, required=True, help='Path to input embeddings file')
    parser.add_argument('--result_dir', type=str, default='./Vicinity_results', help='parent directory of results')
    parser.add_argument('--save_results', action='store_true', help='Flag to save results')
    parser.add_argument('--compute_LD', action='store_true', help='Flag to compute LD')
    parser.add_argument('--plot_results', action='store_true', help='Flag to generate plots')
    parser.add_argument('--parallel', action='store_true', help='parallelize KNN search , suggested to use with more than 500k seqs')
    parser.add_argument('--chosen_metric', type=str, choices=['cosine', 'euclidean'], default='cosine', help='Metric to use')
    parser.add_argument('--sample_size', type=int, default= 0 , help='Size of the max sample of each label')
    parser.add_argument('--LD_sample_size', type=int, default= 10000 , help='Number of seqs (X vs ALL) to check in the LD calculations')
    parser.add_argument('--precomputed_LD', type=str, required=False,help='path of the precomputed file.csv with LD results' ,default="") #Old to be removed
    parser.add_argument("--radius_range", type=str, default="7,24,1", help="Specify the min and max radius and steps separated by a comma (e.g., '7,24,1')")
    parser.add_argument('--skip_knn', action='store_true', help='Flag to skip the  KNN Vicinity ')
    parser.add_argument('--LD_matrix', type=str, required=False,default="" , help='path of the precomputed file.pt with LD results')
    parser.add_argument('--fix_nan_inf', action='store_true',default="True", help='Fix NaN and inf values in the embeddings, by replacing them with 0')
    return parser.parse_args()


def create_result_folder(res_folder):
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    else:
        print(f"Warning: Result folder {res_folder} already exists. Output files may be overwritten.")


def load_data(input_metadata, input_embeddings,idx_reference):
    if not os.path.exists(input_metadata):
        raise FileNotFoundError(f"Metadata file not found: {input_metadata}")
    if not os.path.exists(input_embeddings):
        raise FileNotFoundError(f"Embeddings file not found: {input_embeddings}")
    
    tensors = np.load(input_embeddings, mmap_mode='r')
    # tensors = torch.load(input_embeddings).numpy()  
    seqs = pd.read_csv(input_metadata, sep=None , engine ='python')
    # Check for NaN values in seqs and convert to "unkn"
    print("Checking for NaN values in the sequences dataframe...")
    # Check if the dataframe contains any NaN values
    if seqs.isna().any().any():
        print(f"Found NaN values in the dataframe.")
        # Check specifically for NaN values in the binding label column
        if args.df_affinity_colname in seqs.columns and seqs[args.df_affinity_colname].isna().any():
            print(f"Found {seqs[args.df_affinity_colname].isna().sum()} NaN values in the binding label column '{args.df_affinity_colname}'. Converting them to 'unkn'...")
            seqs[args.df_affinity_colname] = seqs[args.df_affinity_colname].fillna("unkn")
        # Fill NaN values in other columns as well
        seqs = seqs.fillna("unkn")
    else:
        print("No NaN values found in the sequences dataframe.")
    if idx_reference == "":
        seqs['id'] = np.arange(0, len(seqs))
        tensors_df = pd.DataFrame({
            'id': np.arange(0, len(tensors)),
            'embedding': list(tensors)
        })
        df = pd.merge(seqs, tensors_df, on='id')
    else:
        #'/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis/antiberta2/full_chain/100k_sample_trastuzmab_full_chain_antiberta2_idx.csv'
        #'/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv'
        idx_df= pd.read_csv(idx_reference, sep =None , engine ='python')
        print(idx_df.head())
        print(tensors[:5])
        print(seqs.head())
        tensors_df = pd.DataFrame({
            'tensor_id': idx_df['index'],
            'sequence_id' : idx_df['sequence_id'],
            # 'embedding': list(tensors)
        })
        df = pd.merge(seqs, tensors_df, on='sequence_id')
        # embeddings = tensors[[df['tensor_id'].values]]  # extracting the correct entries matching the sampled indices
        # print(embeddings)
    print("...Removing duplicated sequences ...")
    df = df[~df[args.df_junction_colname].duplicated(keep=False)]
    print(f"Number of sequences in the dataset after de-deuplication: {len(df)}, per class: {df[args.df_affinity_colname].value_counts()}")
    embeddings = tensors[df['tensor_id'].values]
    df = df.reset_index(drop=True)
    df['id'] = np.arange(0, len(df))

    return df , embeddings



def filter_data(df, max_junction_length=40, sample_size=10000,
                rand_seed=123, junction_aa_col='junction_aa', affinity_col='affinity'):
    try:
        df['junction_length'] = df[junction_aa_col].apply(len)
        df = df[df['junction_length'] <= max_junction_length]
    except KeyError:
        print("Error: No junction aa has been provided, or the colname is wrong")
        return pd.DataFrame()
    unique_labels = df[affinity_col].unique()
    affinity_dfs = {}
    for label in unique_labels:
        print(f"Sampling {label} ...")
        curr_sample_size = min(len(df[df[affinity_col] == label]), sample_size)
        filtered_df = df[df[affinity_col] == label].sample(n=curr_sample_size, random_state=rand_seed)
        affinity_dfs[label] = filtered_df
        print(affinity_dfs[label] )
    return pd.concat(affinity_dfs.values(), ignore_index=True)

def format_density(value, precision):
        return f"{value:.{precision}f}".replace(".", "")

#function to sample the max numb of sequences if they are below the chosen sample size
def sample_affinities(df, sample_size, df_affinity_colname ='affinity'):
    np.random.seed(123)
    df['affinity']=df[df_affinity_colname]
    # Check the maximum available samples for each affinity
    count_per_affinity = df['affinity'].value_counts()
    # Sample indices for each affinity based on the minimum of available count or the desired sample size
    sampled_indices = pd.Index([])
    for affinity, count in count_per_affinity.items():
        print(affinity)
        print(count)
        current_sample_size = min(sample_size, count)  # Adjust the sample size if necessary
        sampled_indices = sampled_indices.append(df[df['affinity'] == affinity].sample(n=current_sample_size, random_state=42).index)
    
    return df.loc[sampled_indices]


def convert_ld_results_to_long_format(ld_df):
    """
    Converts the input DataFrame from wide to long format.
    
    Parameters:
    -----------
    ld_df : pandas.DataFrame
        The input DataFrame containing columns:
            - 'LD_thr': The threshold value.
            - 'vicinity': A list of percentages.
            - 'num_points': A list of neighbor counts.
            - 'binding_labels': A list of binding labels.
            - 'sample_id': A list of sample IDs.
    
    Returns:
    --------
    new_df : pandas.DataFrame
        A new DataFrame in long format with columns:
        'ID', 'Threshold', 'Affinity', 'Percentage', 'Neighbors', 'local_index'
    """
    rows_list = []
    for _, row in ld_df.iterrows():
        threshold = row['LD_thr']
        # Extract lists from the current row.
        vicinity_all = row.get('vicinity', [])
        num_points_all = row.get('num_points', [])
        binding_labels = row.get('binding_labels', [])
        sample_id = row.get('sample_id', None)
        # Convert to list if possible.
        if hasattr(sample_id, 'tolist'):
            sample_id = sample_id.tolist()
        elif sample_id is None:
            sample_id = []
        else:
            sample_id = list(sample_id)
        
        # Determine the number of elements in the lists.
        n_elements = len(vicinity_all)
    
        for i in range(n_elements):
            rows_list.append({
                'ID': sample_id[i] if i < len(sample_id) else None,
                'Threshold': threshold,
                'Affinity': binding_labels[i] if i < len(binding_labels) else None,
                'Percentage': vicinity_all[i] if i < len(vicinity_all) else None,
                'Neighbors': num_points_all[i] if i < len(num_points_all) else None,
                'local_index': i,
            })
    
    new_df = pd.DataFrame(rows_list)
    return new_df


def adjacency_to_edgelist(matrix, matching_indices_df, output_file=None):
    """
    Convert a sparse adjacency matrix to an edge list and optionally save it to a CSV file.
    
    Parameters:
    -----------
    matrix : scipy.sparse.csr_matrix
        The adjacency matrix to convert.
    matching_indices_df : pandas.DataFrame
        DataFrame containing the mapping between matrix indices and sequence IDs.
        Must have a 'sequence_id' column.
    output_file : str, optional
        If provided, the edge list will be saved to this CSV file path.
    
    Returns:
    --------
    edges : set
        A set of tuples, where each tuple contains the source and target sequence IDs.
    """
    # Get the non-zero elements (edges) from the sparse matrix
    rows, cols = matrix.nonzero()
    
    # Map row and column indices to sequence IDs
    rows = matching_indices_df['sequence_id'].values[rows]
    cols = matching_indices_df['sequence_id'].values[cols]
    
    # Create a set of edges (sorted to avoid duplicates)
    edges = set()
    for i, j in zip(rows, cols):
        if i != j:
            edges.add(tuple(sorted((i, j))))
    
    # If output file is specified, save the edge list to CSV
    if output_file:
        with open(output_file, "w") as f:
            _ = f.write("source,target\n")
            for edge_a, edge_b in edges:
                _ = f.write(f"{edge_a},{edge_b}\n")
    print(f"edge list saved to {output_file}")


def adjacency_to_csv(matrix, filename):
    """
    Save the adjacency matrix to a CSV file.
    
    Parameters:
    -----------
    matrix : scipy.sparse.csr_matrix
        The adjacency matrix to save.
    filename : str
        The name of the output CSV file.
    """
    # Convert the sparse matrix to a dense format and then to a DataFrame
    neighbors_counts = matrix.getnnz(axis=1)
    df = pd.DataFrame({
        "row": id_index_sample,
        "Neighbors_Count": neighbors_counts
    })
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

def check_and_fix_embeddings(embeddings, fix_nan_inf=True):
    import sklearn.utils.validation

    # Check for inf values in the embeddings
    if np.isinf(embeddings).any():
        inf_count = np.isinf(embeddings).sum()
        if inf_count > 0:
            inf_indices = np.argwhere(np.isinf(embeddings))
            unique_cols = np.unique(inf_indices[:, 1])
            print(f"Found {inf_count} inf values in the embeddings at indices (row, col): {inf_indices[:10]},\nunique columns: {unique_cols}")
            if inf_count > 10:
                print(f"... (showing first 10 of {inf_count} infs)")
        if fix_nan_inf:
            embeddings[np.isinf(embeddings)] = 0
            print("ERROR! Infinite values found in the embeddings. Replacing them with 0.")
        else:
            raise ValueError("Embeddings contain infinite values. Please check your input data.")

    # Check for nan values in the embeddings
    if np.isnan(embeddings).any():
        nan_count = np.isnan(embeddings).sum()
        if nan_count > 0:
            nan_indices = np.argwhere(np.isnan(embeddings))
            unique_cols = np.unique(nan_indices[:, 1])
            print(f"Found {nan_count} NaN values in the embeddings at indices (row, col): {nan_indices[:10]},\nunique columns: {unique_cols}")
            if nan_count > 10:
                print(f"... (showing first 10 of {nan_count} nans)")
        if fix_nan_inf:
            embeddings[np.isnan(embeddings)] = 0
            print("ERROR! NaN values found in the embeddings. Replacing them with 0.")
        else:
            raise ValueError("Embeddings contain NaN values. Please check your input data.")
    return embeddings

import numpy as np


import numpy as np

def select_first_d_dims(embeddings, token_dim=1280, num_cols=50):
    """
    Select only the first `num_cols` dimensions out of `token_dim` per token.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, T*token_dim)
        Flattened embeddings per protein.
    token_dim : int, default=1280
        Dimensionality of each token embedding.
    num_cols : int, default=50
        Number of dimensions per token to keep.

    Returns
    -------
    np.ndarray, shape (N, T*num_cols)
        Flattened embeddings with only the first `num_cols` dims kept per token.
    """
    N, flat_dim = embeddings.shape
    T = flat_dim // token_dim  # number of tokens per protein
    
    idx = np.arange(T)[:, None] * token_dim + np.arange(num_cols)
    idx = idx.ravel()
    print(f" idx chosen {idx}")
    
    return embeddings[:, idx]



args = parse_arguments()


print("Starting analysis ...")


analysis_name = args.analysis_name
result_folder= os.path.join( args.result_dir, f"{analysis_name}/")
df_junction_colname= args.df_junction_colname
df_affinity_colname= args.df_affinity_colname
idx_reference=args.input_idx
chosen_metric = args.chosen_metric
parallel_choice =args.parallel
skip_knn = args.skip_knn

try:
  min_radius, max_radius, step = float(args.radius_range.split(',')[0]), float(args.radius_range.split(',')[1]), float(args.radius_range.split(',')[2])
  #ED_radius = range(min_radius, max_radius )
  ED_radius= np.arange(min_radius, max_radius, step)
#   print("Euclidean distance radius range and steps:", list(ED_radius))
except ValueError:
  print("Error: Please ensure you provide two integers separated by a comma for the radius range.")


print(f" this is the result folder {result_folder}")

create_result_folder(result_folder)



# df = load_data(args.input_metadata, args.input_embeddings, idx_reference)
# tensors = np.load('test_memmap2.npy', mmap_mode='r')
# df , embeddings  = load_data(args.input_metadata, 'test_memmap2.npy', idx_reference)
df , embeddings  = load_data(args.input_metadata, args.input_embeddings, idx_reference)
id_index_sample = df['id']


embeddings = select_first_d_dims(embeddings, token_dim=1280, num_cols=300)
print(embeddings.shape)


if args.sample_size != 0 :
    df_sample_filt = filter_data(df,
                                    sample_size= args.sample_size,
                                    junction_aa_col=args.df_junction_colname,
                                    affinity_col=args.df_affinity_colname)
    id_index_sample = df_sample_filt['id']
    embeddings = embeddings[df_sample_filt['id'].values]
    print(f"Number of sequences in the dataset: {len(df_sample_filt)}")


id_index_sample_df = id_index_sample.reset_index().rename(columns={'index': 'vicinity_index'})
matching_indices_df = id_index_sample_df.merge(df[['id', 'sequence_id']], on='id', how='left')
matching_indices_df.columns = ['vicinity_index', 'metadata_id', 'sequence_id']
matching_indices_df.to_csv(f"{result_folder}_index_mapping_{analysis_name}.csv", index=False)
 


# Call the function after loading embeddings
embeddings = check_and_fix_embeddings(embeddings, fix_nan_inf=args.fix_nan_inf)

"""  ** Run Vicinity analysis ** """

max_neighbors = 2000 # This is the maximum number of neighbors you're interested in
part1 = np.arange(2, 304, 4)  # check in detail first 300 NN
part2 = np.arange(350, max_neighbors+1, 50)  # Second part: numbers from 300 to 1000 with steps of 50
neighbor_numbers = np.concatenate((part1, part2))

#neighbor_numbers = np.arange(2, 30, 4) #for debug purposes

#KNN vicinity
vicinity_analysis_instance = Vicinity_analysis(df,
                                                embeddings,
                                                neighbor_numbers,
                                                id_index_sample,
                                                colname_affinity=df_affinity_colname,
                                                colname_junction=df_junction_colname,
                                                metric= chosen_metric,
                                                parallel= parallel_choice,
                                                skip_KNN=skip_knn)
vicinity_analysis_instance.run_analysis()       # This populates the necessary attributes

# vicinity_analysis_instance.label_results

print(df[df_affinity_colname].value_counts())

# Run Vicinity Radius
percentages_results,perc_df, res_df, mean_num_points, LD1_res, LD2_res = vicinity_analysis_instance.perc_Euclidian_radius(ED_radius)
tmp_ed_sum=vicinity_analysis_instance.summary_results 
# Run Adjacency matrix
# density_thresholds= [0.001,0.002, 0.003, 0.004,0.005, 0.006, 0.007, 0.008, 0.009, 0.01]

knn_density_thr=[10, 50, 100, 500, 1000]
knn_density_thr=[10]
density_thresholds=[]
for i in knn_density_thr:
    density_thresholds.append(vicinity_analysis_instance.NN_dist[:,i].mean())
    print(f"Computed density  thresholds knn = {knn_density_thr}: {density_thresholds}")

lin0_density_thr=[vicinity_analysis_instance.lin_density_thresholds.tolist()[0]]
#7 evenly spaced thresholds acrosse the 0.01 and 0.95 quantiles of the distances


LD_thr =[]
for thr in [1,2]:
    mask = (vicinity_analysis_instance.NN_lev[:, 1:] <= thr)
    mean_val = np.mean(vicinity_analysis_instance.NN_dist[:, 1:][mask])
    print(f"Mean NN_dist value where NN_lev <= {thr}:", mean_val)
    LD_thr.append(mean_val)


density_thresholds = density_thresholds + lin0_density_thr + LD_thr  #choose the lin0 - the first threshold
print(f"Computing adjancey for the following density thresholds: {density_thresholds}")
adjacency_matrices, row_ids = vicinity_analysis_instance.compute_adjacency_matrices( density_thresholds )
 
# Save the results
ED_filename= f"{result_folder}summary_results_ED_{analysis_name}.csv"
    
vicinity_analysis_instance.summary_results.to_csv(ED_filename)
perc_df.to_csv(f"{result_folder}_raw_perc_ED_{analysis_name}.csv")
for i, matrix in enumerate(adjacency_matrices):
    print(f"Adjacency matrix {i} shape: {matrix.shape}")
    # Format the current density to 4 decimals without the decimal point
    reformat_density = format_density(density_thresholds[i], precision=4)

    # Check if the next density value (if it exists) formats to the same string at 4 decimal places.
    if i + 1 < len(density_thresholds) and reformat_density == format_density(density_thresholds[i+1], precision=4):
        # Increase precision to 5 if needed.
        reformat_density = format_density(density_thresholds[i], precision=5)
    if i < len(knn_density_thr):
        nn_info = knn_density_thr[i]
        matrix_name = f"{result_folder}adjacency_matrix_NN{nn_info}_R{reformat_density}_{analysis_name}.npz"
        adjacency_to_csv(matrix, f"{result_folder}neighbors_diag_NN{nn_info}_R{reformat_density}_{analysis_name}.csv")
        save_npz(matrix_name, matrix)
    elif i < len(knn_density_thr) + len(lin0_density_thr):
        lin_info = i - len(knn_density_thr)
        matrix_name = f"{result_folder}adjacency_matrix_lin{lin_info}_R{reformat_density}_{analysis_name}.npz"
        adjacency_to_csv(matrix, f"{result_folder}neighbors_diag_lin{lin_info}_R{reformat_density}_{analysis_name}.csv")
        save_npz(matrix_name, matrix)
    else:
        ld_info = i - len(knn_density_thr) - len(lin0_density_thr)
        matrix_name = f"{result_folder}adjacency_matrix_LD{ld_info}_R{reformat_density}_{analysis_name}.npz"
        adjacency_to_csv(matrix, f"{result_folder}neighbors_diag_LD{ld_info+1}_R{reformat_density}_{analysis_name}.csv")
        save_npz(matrix_name, matrix)
    
    print(f"Adjacency matrix saved at {matrix_name}")
# save index mapping

# Merge the 'id' and 'sequence_id' columns from the vicinity analysis DataFrame with the sampled id indices,
# then save the merged DataFrame as a CSV file for later reference.
pd.merge(vicinity_analysis_instance.df[['id','sequence_id']],
         pd.DataFrame({'id': id_index_sample.reset_index(drop=True)}),
         on='id', how='inner').to_csv(f"{result_folder}index_sequence_id_{analysis_name}.csv", index=False)

# adjacency_matrices[2] is the LD-1 threshold
adjacency_to_edgelist(adjacency_matrices[2], matching_indices_df, output_file=f"{result_folder}edges_list_avgLD1_{analysis_name}.csv") # adjacency_matrices[2] is the LD-1 threshold



# ----------------- Save vicinity results Pickle ------
if args.save_results:
    vicinity_analysis_instance.save_to_pickle(f"{result_folder}Vicinity_{analysis_name}.pkl")


#------ compute the LD distance on a substet as comparator
np.random.seed(123)
#function to sample the max numb of sequences if they are below the chosen sample size



print("LD calculation...")

chosen_sample_size=  args.LD_sample_size
# chosen_sample_size=  50000 #debug
# chosen_sample_size=  25000 #debug
LD_filename=f"{result_folder}d_mean1_summary_LD_{analysis_name}_{args.sample_size}k.csv"

if args.LD_matrix != "":
    print( "Using the LD matrix file at ", args.LD_matrix)
    LD_filename=f"{result_folder}d_mean1_summary_LD_{analysis_name}_{(len(id_index_sample))/1000}k.csv"
    matrix_path = args.LD_matrix
    d_res1,d_mean1 = prepare_data_for_plotting_LD_MAT(
        matrix_path,
        pd.read_csv(args.input_metadata, sep=None),
        vicinity_analysis_instance.df,
        # "all",
        id_index_sample,
        max_LD=7,
        junction_aa_col=args.df_junction_colname,
        affinity_col=args.df_affinity_colname
    )
    d_mean1.to_csv(LD_filename)
    d_res1_long = convert_ld_results_to_long_format(d_res1)
    d_res1_long.to_csv(f"{result_folder}d_whole_LD_stats_{analysis_name}_{(len(id_index_sample))/1000}k.csv")



if args.compute_LD == True:
    print('LD computing...')
    rand_100k=sample_affinities(pd.read_csv(args.input_metadata, sep=None), chosen_sample_size, df_affinity_colname).index
    max_LD=5
    d_res1,d_mean1 = prepare_data_for_plotting( pd.read_csv(args.input_metadata, sep=None),max_LD, sampled_indices=rand_100k) # to get VICINITY percentages of LD dist 
    #( for i in sampled_indices --> LD calculation  i vs ALL)
    #save_to_pickle(d_mean1, LD_filename)
    d_mean1.to_csv(LD_filename)
# TODO ---- Please parallelize this function, it's very slow, at least run each LD threshold on a different core



if args.plot_results== True:
    if args.compute_LD == False:
        LD_filename = args.precomputed_LD
    else:
        print("Using the recently computed LD files")
    
    run_ggplot_vicinity(analysis_name,ED_filename,LD_filename, output_path= result_folder )



with open(f"{result_folder}param_log.txt", 'w') as f:
        f.write("Parameters Summary:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")







