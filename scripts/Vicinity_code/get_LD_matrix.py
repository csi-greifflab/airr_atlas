import torch
import os
import pandas as pd
import numpy as np


from rapidfuzz.distance import Levenshtein as RapidfuzzLevenshtein
import numpy as np
from rapidfuzz.distance import Levenshtein
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

import numpy as np
from rapidfuzz.distance import Levenshtein
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
# from Vicinity_pipeline import load_data, filter_data 
from tqdm import tqdm

from rapidfuzz.distance import Levenshtein

# from Vicinity_pipeline import load_data, filter_data 
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

import torch
from rapidfuzz.distance import Levenshtein
from joblib import Parallel, delayed


#     '--input_metadata' ,'/doctorai/niccoloc/trastuzumab_metadata.csv' ,
#     '--input_embeddings' ,'/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab/antiberta2/cdr3_only/100k_sample_trastuzmab_cdr3_heavy_only_antiberta2_layer_16.pt' ,
import pandas as pd
data= pd.read_csv('/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv', sep='\t')
data= pd.read_csv('/doctorai/userdata/airr_atlas/data/sequences/bcr/porebski_metadata.csv')
data['sequence_id']


def compute_pairwise_distance_batch_fast(strings, batch_size=100):
    # """
    # Computes pairwise Levenshtein distances in batches and parallelizes the computation.
    # 
    # Args:
    #     strings (list of str): List of strings to compute distances for.
    #     batch_size (int): Size of each batch for parallel processing.
    # 
    # Returns:
    #     torch.Tensor: A tensor containing the pairwise distances (int16).
    # """
    num_strings = len(strings)
    # Use NumPy array for intermediate calculations
    distance_matrix = np.zeros((num_strings, num_strings), dtype=np.int8)
    def compute_upper_triangular(start, end):
        batch_distances = []
        for i in range(start, end):
            row_distances = []
            for j in range(i, num_strings):
                distance = Levenshtein.distance(strings[i], strings[j])
                row_distances.append((i, j, distance))
            batch_distances.extend(row_distances)
        return batch_distances
    # Parallelize computation of upper triangular matrix
    num_batches = (num_strings + batch_size - 1) // batch_size
    from tqdm_joblib import tqdm_joblib 
    with tqdm_joblib(desc="Computing pairwise distances", total=num_batches) as progress_bar:
        results = Parallel(n_jobs=30)(
        delayed(compute_upper_triangular)(i * batch_size, min((i + 1) * batch_size, num_strings))
        for i in range(num_batches)
    )
    # Fill in the upper triangular matrix
    for batch in results:
        for i, j, distance in batch:
            distance_matrix[i, j] = distance
    # Fill in the lower triangular matrix (symmetry)
    i_lower = np.tril_indices(num_strings, -1)
    distance_matrix[i_lower] = distance_matrix.T[i_lower]
    print("Done")
    # Convert to PyTorch tensor
    return torch.tensor(distance_matrix, dtype=torch.int16)

def extract_subsample(distance_matrix, indexes):
    indexes_tensor = torch.tensor(indexes, dtype=torch.long)
    subsample = distance_matrix[indexes_tensor][:, indexes_tensor]
    return subsample


def compute_full_distance_matrix_memmap(strings, batch_size=1000, filename='distance_matrix.dat'):
    # """
    # Computes the full pairwise Levenshtein distance matrix in batches,
    # stores it on disk using a memory-mapped file, and displays a progress bar.
    # 
    # Args:
    #     strings (list of str): List of all sequences.
    #     batch_size (int): Number of rows to process in each batch.
    #     filename (str): Filename for the memory-mapped file.
    # 
    # Returns:
    #     np.memmap: Memory-mapped distance matrix.
    # """
    num_strings = len(strings)
    shape = (num_strings, num_strings)
    # Create a memory-mapped file for the distance matrix
    distance_matrix = np.memmap(filename, dtype=np.int8, mode='w+', shape=shape)
    # Initialize the distance matrix to zeros
    distance_matrix[:, :] = 0
    # Function to compute distances for a batch of rows
    def compute_batch(start_row, end_row):
        for i in range(start_row, end_row):
            row_distances = np.zeros(num_strings - i, dtype=np.int8)
            for j in range(i, num_strings):
                distance = Levenshtein.distance(strings[i], strings[j])
                row_distances[j - i] = distance
            # Write the computed distances to the upper triangular part
            distance_matrix[i, i:num_strings] = row_distances
            # Note: Do not update the symmetric lower triangle here to avoid race conditions
    # Compute the number of batches
    num_batches = (num_strings + batch_size - 1) // batch_size
    # Use tqdm_joblib to display a progress bar with Parallel processing
    with tqdm_joblib(desc="Computing pairwise distances", total=num_batches) as progress_bar:
        Parallel(n_jobs=30)(
            delayed(compute_batch)(i * batch_size, min((i + 1) * batch_size, num_strings))
            for i in range(num_batches)
        )
    # After computing the upper triangular matrix, fill the lower triangle
    i_lower = np.tril_indices(num_strings, -1)
    distance_matrix[i_lower] = distance_matrix.T[i_lower]
    # Flush changes to disk
    distance_matrix.flush()
    return distance_matrix

 

def compute_levenshtein(ref_seq, sequences):
    return [RapidfuzzLevenshtein.distance(ref_seq, seq) for seq in sequences]


strings = ["apple", "apply", "banana", "bandana", "cherry"]
strings = data['cdr3_aa'][:10000]
strings = data['cdr3_aa']
strings = data['cdr3']


compute_levenshtein(strings[1], strings[1:])

current_seq_id = ["tz_heavy_101", "tz_heavy_103", "tz_heavy_109"]
index = data[data['sequence_id'].isin(current_seq_id)].index
index
initial_seq = data.loc[index, 'cdr3_aa']

indexes = [0, 2, 4]  # Subsample indexes

 

# Check if the two distance matrices are identical
if torch.equal(distance_matrix_s, distance_matrix_p):
    print("The two distance matrices are identical.")
else:
    print("The two distance matrices are not identical.")
 
subsample_p = extract_subsample(distance_matrix_p, index)

# Check if the two subsamples are identical
if torch.equal(subsample_s, subsample_p):
    print("The two subsamples are identical.")

 


lev_dists= distance_matrix[index][:, index]
lev_dist_orig =compute_levenshtein(strings[1], strings[:50])

indices_at_dist = [i for i, x in enumerate(lev_dists) if 0 < x <= 3]

indices_at_dist = np.where((lev_dists > 0) & (lev_dists <= 3))



distance_matrix = compute_full_distance_matrix_memmap(strings, batch_size=500, filename='porebski_LD_matrix.dat')
distance_matrix = compute_pairwise_distance_batch_fast(strings,  batch_size=500)


# np.save('porebski_LD_matrix.npy', distance_matrix)

current_directory = os.getcwd()
print(f"The current directory is: {current_directory}")


distance_matrix.shape


dist_torch=torch.tensor(distance_matrix, dtype=torch.int8)
dist_torch[:4,:4]
torch.save(dist_torch, 'tz_LD_dist_mat_HB_LB.pt')

x1=torch.load('/doctorai/niccoloc/tz_LD_dist_mat_HB_LB.pt')


extract_subsample(x1, [0, 1, 2])

compute_levenshtein(strings[0], strings[:3])

data

data
x1.shape

def load_data(input_metadata, input_embeddings,idx_reference):
    if not os.path.exists(input_metadata):
        raise FileNotFoundError(f"Metadata file not found: {input_metadata}")
    if not os.path.exists(input_embeddings):
        raise FileNotFoundError(f"Embeddings file not found: {input_embeddings}")
    
    tensors = torch.load(input_embeddings).numpy()    
    seqs = pd.read_csv(input_metadata, sep=None , engine ='python')
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
        tensors_df = pd.DataFrame({
            'tensor_id': idx_df['index'],
            'sequence_id' : idx_df['sequence_id'],
            'embedding': list(tensors)
        })
        df = pd.merge(seqs, tensors_df, on='sequence_id')
    print("...Removing duplicated sequences ...")
    # df = df[~df[args.df_junction_colname].duplicated(keep=False)]
    # df = df[~df[args.df_junction_colname].duplicated(keep=False)]
    df = df.reset_index(drop=True)
    df['id'] = np.arange(0, len(df))
    return df


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
        curr_sample_size = min(len(df[df[affinity_col] == label]), sample_size)
        filtered_df = df[df[affinity_col] == label].sample(n=curr_sample_size, random_state=rand_seed)
        affinity_dfs[label] = filtered_df
    return pd.concat(affinity_dfs.values(), ignore_index=True)



id_index_sample = df_sample_filt['id']
print(f"Number of sequences in the dataset: {len(df_sample_filt)}")


data[data['sequence_id'] == 'tz_heavy_21118']


data[data['sequence_id'].isin(vicinity_analysis_instance.df['sequence_id'])]

data.loc[401]
data.loc[695]

ids= [1,788,2,3,900]
df=vicinity_analysis_instance.df.iloc[ids]
df=vicinity_analysis_instance.df.loc[id_index_sample]

df
matrix_indexes = data[data['sequence_id'].isin(df['sequence_id'])]

# Match the matrix indexes with the binding labels
matrix_indexes = data[data['sequence_id'].isin(df['sequence_id'])]
binding_labels = matrix_indexes['binding_label'].values



lev_dists = extract_subsample(x1, matrix_indexes.index)

compute_levenshtein(strings[matrix_indexes.index[0]], strings[ matrix_indexes.index])


# Map string variables to numerical variables
affinity_mapping = {'lb': 0, 'hb': 1}
data['binding_label_num'] = data['binding_label'].map(affinity_mapping)

# Extract subsample using numerical labels
affinity_labels = extract_subsample(torch.tensor(data['binding_label_num'].values), matrix_indexes.index)

compute_levenshtein(strings[11478], strings[[11478, 19204, 60885, 83611, 105357]])

def extract_subsample(distance_matrix, indexes):
    indexes_tensor = torch.tensor(indexes, dtype=torch.long)
    subsample = distance_matrix[indexes_tensor][:, indexes_tensor]
    return subsample

# Create a matrix from binding labels
def create_binding_label_matrix(binding_labels):
    num_labels = len(binding_labels)
    binding_label_matrix = np.zeros((num_labels, num_labels), dtype=np.int8)
    for i in range(num_labels):
        for j in range(num_labels):
            binding_label_matrix[i, j] = binding_labels[j]
    return torch.tensor(binding_label_matrix, dtype=torch.int8)

binding_label_matrix = create_binding_label_matrix(df['binding_label_num'])

def count_values_greater_than_X(tensor, ld_thr, labels):
    # Convert tensor to numpy array if it's a PyTorch tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    # Create a boolean mask where condition is satisfied
    condition = (0 < tensor) & (tensor <= ld_thr)
    # Retrieve column indices for each row where the condition holds true
    indices_per_row = [np.where(row_mask)[0] for row_mask in condition]
    res = []
    num_points = []
    for i, idx in enumerate(indices_per_row):
        if len(idx) > 0:
            res.append(sum(labels[idx] == labels[i]) / len(idx))
            num_points.append(len(idx))
        else:
            res.append(np.nan)
            num_points.append(0)
    return res, num_points

def group_results_by_binding_label(results, labels):
    grouped_results = {}
    for result, label in zip(results, labels):
        if label not in grouped_results:
            grouped_results[label] = []
        grouped_results[label].append(result)
    return grouped_results

# Example usage
tensor = torch.tensor([[0, 2, 3, 7], [1, 1, 1, 7], [4, 5, 6, 7]])





tensor = lev_dists
results = []
for LD_thr in range(1, 3):
    vicinity, num_points = count_values_greater_than_X(tensor, LD_thr, binding_labels)
    vicinity_by_class = group_results_by_binding_label(vicinity, binding_labels)
    num_points_by_class = group_results_by_binding_label(num_points, binding_labels)
    results.append((LD_thr, vicinity, vicinity_by_class, num_points, num_points_by_class))

    # Create a DataFrame for the results
# Create a DataFrame for the results
results_df = pd.DataFrame(results, columns=['LD_thr', 'vicinity', 'vicinity_by_class', 'num_points', 'num_points_by_class'])
results_df



# Initialize lists to store the summary statistics
mean_vicinity = []
mean_num_points = []
perc_nan = []
mean_num_points_HB = []
mean_num_points_LB = []
mean_vicinity_HB = []
mean_percentage_LB = []
perc_nan_HB = []
perc_nan_LB = []

# Calculate summary statistics for each LD threshold
# for LD_thr, counts, vicinity_by_class in results[0]:

LD_thr= results[0][0]
vicinity= results[0][1]
vicinity_by_class= results[0][2]
num_points= results[0][3]
num_points_by_class = results[0][4]


for LD_thr, vicinity, vicinity_by_class, num_points, num_points_by_class in results:
    # Calculate mean percentage and number of points
    vicinity_scores = [count for count in vicinity if not np.isnan(count)]
    mean_vicinity.append(np.mean(vicinity_scores))
    mean_num_points.append(np.mean(num_points))
    perc_nan.append(np.isnan(vicinity).mean())

    # Calculate statistics for Hb group
    hb_counts = vicinity_by_class.get('hb', [])
    num_points_HB = num_points_by_class.get('hb', [])

    vicinity_score_HB = [count for count in hb_counts if not np.isnan(count)]
    mean_vicinity_HB.append(np.mean(vicinity_score_HB))
    mean_num_points_HB.append(np.mean(num_points_HB))
    perc_nan_HB.append(np.isnan(hb_counts).mean())

    # Calculate statistics for Lb group
    lb_counts = vicinity_by_class.get('lb', [])
    num_points_LB = num_points_by_class.get('lb', [])

    vicinity_score_LB = [count for count in lb_counts if not np.isnan(count)]
    mean_percentage_LB.append(np.mean(vicinity_score_LB))
    mean_num_points_LB.append(np.mean(num_points_LB))
    perc_nan_LB.append(np.isnan(lb_counts).mean())

# Threshold,Mean_Percentage,Mean_LD1,Mean_LD2,Mean_Num_Points,Percentage_Null,Perc_lb,AvgPoints_lb,NULLPerc_lb,LD_avgSim_lb

# Create a DataFrame with the summary statistics
summary_df = pd.DataFrame({
    'Threshold': [LD_thr for LD_thr, _, _, _ , _ in results],

    'Mean_Num_Points': mean_num_points,    
    'Mean_Percentage': mean_vicinity,
    'Percentage_Null': perc_nan,

    'Perc_hb': mean_num_points_HB,
    'AvgPoints_hb': mean_vicinity_HB,
    'NULLPerc_hb': perc_nan_HB,

    'Perc_lb': mean_percentage_LB,
    'AvgPoints_lb': mean_num_points_LB,
    'NULLPerc_lb': perc_nan_LB
})

# Display the DataFrame
print(summary_df)

# Display the DataFrame
print(results_df) 


# # Combine results and num_of_points into a single DataFrame
#     columns = [f'LD_{i}' for i in range(1, LD_dist + 1)]
#     df_results = pd.DataFrame(results, columns=[f'Perc_{col}' for col in columns]) # PERCENTAGE OF Points with SAME LABLE (vicinity score)
#     df_num_points = pd.DataFrame(num_of_points, columns=[f'Num_{col}' for col in columns])
#     print(df_results)
#     print(df_num_points)
#     # Merge into one DataFrame
#     affinities = df.loc[sampled_indices, 'affinity']
#     df_combined = pd.concat([df_results, df_num_points], axis=1)
#     df_combined['sample_id'] = sampled_indices
#     df_combined['affinity'] = affinities
#     # Combine results and num_of_points into a single DataFrame
#     columns = [f'LD_{i}' for i in range(1, LD_dist + 1)]
#     df_combined = pd.DataFrame({
#         **{f'Perc_{col}': results[:, idx] for idx, col in enumerate(columns)},
#         **{f'Num_{col}': num_of_points[:, idx] for idx, col in enumerate(columns)},
#         'sample_id': sampled_indices,
#         'affinity': affinities
#     })
    
#     # Calculate summary statistics
#     df_summary = pd.DataFrame()
#     for col in columns:
#         df_combined[f'NaN_Count_{col}'] = df_combined[f'Perc_{col}'].isna()
#         summary_stats = df_combined.groupby('affinity')[[f'Num_{col}', f'NaN_Count_{col}']].agg({
#             f'Num_{col}': 'mean',
#             f'NaN_Count_{col}': 'mean'
#         }).rename(columns={f'Num_{col}': f'Avg_Num_{col}', f'NaN_Count_{col}': f'Avg_NaN_Percentage_{col}'})
#         df_summary = pd.concat([df_summary, summary_stats], axis=1)
    
#     grouped = df_combined.groupby('affinity')
#     grouped[[f'Perc_LD_{i}' for i in range(1,LD_dist+1)]].count()
#     grouped[[f'Perc_LD_{i}' for i in range(1,LD_dist+1)]].mean()
#     df_summary[[f'Num_of_LD_{i}' for i in range(1,LD_dist+1)]] = grouped[[f'Perc_LD_{i}' for i in range(1,LD_dist+1)]].count()
#     df_summary[[f'Perc_LD_{i}' for i in range(1,LD_dist+1)]] = grouped[[f'Perc_LD_{i}' for i in range(1,LD_dist+1)]].mean()
    
#     return df_combined, df_summary 


prepare_data_for_plotting_LD_MAT(data, vicinity_analysis_instance.df, id_index_sample, 3, 'cdr3_aa', 'binding_label' )




import torch
import numpy as np
# Set the GPU device to 1 using environment variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Example usage
tz_metadata= pd.read_csv('/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv', sep='\t')
porebski_metadata= pd.read_csv('/doctorai/userdata/airr_atlas/data/sequences/bcr/porebski_metadata.csv')
# Convert 'non-binder' to 'lb' in the binding_label column of porebski_metadata
porebski_metadata['binding_label'] = porebski_metadata['binding_label'].replace('non-binder', 'lb')

porebski_metadata['binding_label'].value_counts()


# Load the LD matrix and convert to float only once
ld_matrix_path = '/doctorai/niccoloc/tz_LD_dist_mat_HB_LB.pt'
ld_matrix = torch.load(ld_matrix_path)
ld_matrix = ld_matrix.numpy()

# Load the porebski matrix
porebski_matrix_path = '/doctorai/niccoloc/porebski_LD_matrix.npy'
porebski_matrix = np.load(porebski_matrix_path)
porebski_matrix =  porebski_matrix
max_value = np.max(porebski_matrix)


# Sample 100k rows and columns from the LD matrix
sample_size = 80000
indices = np.random.choice(ld_matrix.shape[0], sample_size, replace=False)
ld_matrix_sampled = ld_matrix[indices][:, indices].float()
ld_matrix_mean_sampled = ld_matrix_sampled.cuda().mean().item()
ld_matrix_median_sampled = ld_matrix_sampled.cuda().median().item()
ld_matrix_std_sampled = ld_matrix_sampled.cuda().std().item()


# Sample 100k rows and columns from the porebski matrix
indices = np.random.choice(porebski_matrix.shape[0], sample_size, replace=False)
porebski_matrix_sampled = porebski_matrix[indices][:, indices]




# Compute statistics for the LD matrix using numpy converted tensor
ld_matrix_mean = ld_matrix.mean() 
ld_matrix_mean
ld_matrix_median = np.median(ld_matrix)
ld_matrix_median
ld_matrix_std = ld_matrix.std() 

print(f"LD Matrix - Mean: {ld_matrix_mean}, Median: {ld_matrix_median}, Std: {ld_matrix_std}")

# Compute statistics for the porebski matrix (NumPy takes care of vectorized operations)
porebski_matrix_mean = porebski_matrix.mean()
porebski_matrix_median = np.median(porebski_matrix)
porebski_matrix_std = porebski_matrix.std()

print(f"Porebski Matrix - Mean: {porebski_matrix_mean}, Median: {porebski_matrix_median}, Std: {porebski_matrix_std}")

np.sum(porebski_matrix == 0)

# Sample 100k rows and columns from the LD matrix
ld_matrix_sampled= ld_matrix
sample_size = 25000
indices = np.random.choice(ld_matrix.shape[0], sample_size, replace=False)
ld_matrix_sampled = ld_matrix[indices][:, indices]

# Count the number of instances in the matrix that are equal to 1, 2, 3, and 4
counts = {}
for value in range(1, 5):
    counts[value] = np.sum(ld_matrix_sampled == value)/2
    rapport = counts[value] / ld_matrix_sampled.size
    print(f"Number of instances in the TZ matrix equal to {value}: {counts[value]}, rapport: {rapport}")

# counts[1]/              19000000       #100000000000000000000
                        
# counts_porebski[1]/     278218429446952000000000000

counts_porebski = {}
for value in range(1, 18):
    counts_porebski[value] = np.sum(porebski_matrix == value)/2
    rapport = counts_porebski[value] / porebski_matrix.size
    print(f"Number of instances in the POREBSKI matrix equal to {value}: {counts_porebski[value]}, rapport: {rapport}")

    for value, count in counts.items():

        print(f"Number of instances in the TZ matrix equal to {value}: {count}, rapport: {count / ld_matrix_sampled.size}")

 
porebski_matrix.size
ld_matrix_sampled.size
porebski_matrix.shape
ld_matrix_sampled.shape
ld_matrix.shape


from math import comb
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



porebski_matrix
porebski_metadata

compute_levenshtein(porebski_metadata['cdr3'][0], porebski_metadata['cdr3'][:5])

porebski_matrix[0][:5]

hamming_distance_combinations(3,1)

def hamming_distance_combinations(length, distance):
    num_substitutions = 19  # Standard amino acid set minus 1
    return comb(length, distance) * (num_substitutions ** distance)

# Count how much of each Levenshtein Distance (LD) is covered by the TZ matrix
def compute_ld_coverage(matrix, max_ld, sequence_length):
    """
    Compute the LD coverage for a given matrix.

    Args:
        matrix (np.ndarray): The distance matrix.
        max_ld (int): The maximum Levenshtein distance to consider.
        sequence_length (int): The length of the sequences.

    Returns:
        dict: A dictionary with LD as keys and coverage lists as values.
    """
    ld_space = {}
    for ld in range(1, max_ld + 1):
        ld_coverage = []
        # Calculate the number of possible combinations for the given LD
        ld_combs = hamming_distance_combinations(sequence_length, ld)
        print(f"Number of possible combinations for LD{ld}: {ld_combs}")
        for i in range(len(matrix)):
            # Count the total number of instances with the given LD in the current row
            tot_ld = np.sum(matrix[i] == ld)  # these are neighbors at LD distance X
            # Calculate the coverage as the ratio of observed LD instances to possible combinations
            ld_coverage.append(tot_ld / ld_combs)
        # Calculate the average coverage for the current LD
        avg_coverage = np.mean(ld_coverage)
        print(f"Average LD{ld} coverage: {avg_coverage}")
        ld_space[ld] = ld_coverage
    return ld_space

# Sample the matrices based on the binding labels in Porebski
def sample_matrices_based_on_binding_labels(matrix, metadata , sample_size=0   ):
    hb_indices = metadata[metadata['binding_label'] == 'hb'].index
    lb_indices = metadata[metadata['binding_label'] == 'lb'].index
    
    if sample_size > 0:

        hb_sample_indices = np.random.choice(hb_indices, int(sample_size/2), replace=False)
        lb_sample_indices = np.random.choice(lb_indices, int(sample_size/2), replace=False)
    else:
        hb_sample_indices = hb_indices
        lb_sample_indices = lb_indices

    print(f"Number of HB samples: {len(hb_sample_indices)}")
    print(f"Number of LB samples: {len(lb_sample_indices)}")
    
    sampled_matrix_hb = matrix[hb_sample_indices][:, hb_sample_indices]
    sampled_matrix_lb = matrix[lb_sample_indices][:, lb_sample_indices]
    
    return sampled_matrix_hb, sampled_matrix_lb


# # Count how much of each Levenshtein Distance (LD) is covered by the POREBSKI matrix
# LD_space_porebski = {}
# for LD in range(1, 18):
#     LD_coverage = []
#     # Calculate the number of possible combinations for the given LD
#     LD_combs = hamming_distance_combinations(21, LD)
#     print(f"Number of possible combinations for LD{LD}: {LD_combs}")
#     # LD_zero_correction=(1/LD_combs) * 0.00001
#     for i in range(len(porebski_matrix)):
#         # Count the total number of instances with the given LD in the current row
#         tot_LD = np.sum(porebski_matrix[i] == LD)
#         # Calculate the coverage as the ratio of observed LD instances to possible combinations
#         LD_coverage.append(tot_LD / LD_combs)
#     # Calculate the average coverage for the current LD
#     avg_coverage = np.mean(LD_coverage)
#     print(f"Average LD{LD} coverage for POREBSKI matrix: {avg_coverage}")
#     LD_space_porebski[LD] = LD_coverage

# Compute the LD coverage for the TZ matrix
LD_space_TZ = compute_ld_coverage(ld_matrix, 17, 15)
LD_space_TZ_25k = compute_ld_coverage(ld_matrix_sampled, 17, 10)
LD_space_porebski= compute_ld_coverage(porebski_matrix, 17, 21)
LD_space_TZ=LD_space_TZ_25k

hb_pb_matrix, lb_pb_matrix= sample_matrices_based_on_binding_labels(porebski_matrix, porebski_metadata)
hb_tz_matrix, lb_tz_matrix= sample_matrices_based_on_binding_labels(Ye, tz_metadata, sample_size=25000)

LD_space_porebski_HB= compute_ld_coverage(hb_pb_matrix, 17, 21)
LD_space_porebski_LB= compute_ld_coverage(lb_pb_matrix, 17, 21)

LD_space_TZ_HB= compute_ld_coverage(hb_tz_matrix, 17, 10)
LD_space_TZ_LB= compute_ld_coverage(lb_tz_matrix, 17, 10)




# Process the coverage data correctly for plotting
# ld_values = list(LD_space_porebski.keys())
ld_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


def compute_zero_counts(ld_values, ld_space_dict, matrix_lengths):
    """
    Compute the number of zeros grouped by matrix and LD.

    Args:
        ld_values (list): List of Levenshtein distances.
        ld_space_dict (dict): Dictionary containing LD space coverage for different matrices.
        matrix_lengths (dict): Dictionary containing the lengths of the matrices.

    Returns:
        pd.DataFrame: DataFrame containing zero counts and normalized zero counts.
    """
    zero_counts = []
    for ld in ld_values:
        for matrix_name, ld_space in ld_space_dict.items():
            zeros = sum(1 for coverage in ld_space[ld] if coverage == 0)
            zero_counts.append([ld, zeros, matrix_name])
    
    # Create a DataFrame for zero counts
    zero_counts_df = pd.DataFrame(zero_counts, columns=['LD', 'Zero_Count', 'Matrix'])
    for row in zero_counts:
        row.append(row[1] / matrix_lengths[row[2]])
    zero_counts_df['Normalized_Zero_Count'] = [row[3] for row in zero_counts]
    
    return zero_counts_df

def flatten_coverage_data(ld_values, ld_space_dict):
    """
    Flatten lists and pair with corresponding matrix and LD for plotting.

    Args:
        ld_values (list): List of Levenshtein distances.
        ld_space_dict (dict): Dictionary containing LD space coverage for different matrices.

    Returns:
        pd.DataFrame: DataFrame containing flattened coverage data.
    """
    data = []
    for ld in ld_values:
        for matrix_name, ld_space in ld_space_dict.items():
            for coverage in ld_space[ld]:
                if coverage > 0:  # Exclude zero values
                    data.append([ld, coverage, matrix_name])
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame(data, columns=['LD', 'Coverage', 'Matrix'])
    epsilon = 1e-30  # Small positive value
    plot_data['Coverage'] = plot_data['Coverage'].replace(0, epsilon)
    
    return plot_data

# Example usage
ld_values = list(LD_space_porebski.keys())

# Dictionary containing LD space coverage for different matrices
ld_space_dict = {
    "Porebski_hb": LD_space_porebski_HB,
    "Porebski_lb": LD_space_porebski_LB,
    "Tz_25k_hb": LD_space_TZ_HB,
    "Tz_25k_lb": LD_space_TZ_LB
}

zero_try=[]
for matrix_name, ld_space in ld_space_dict.items():
    print(f"Matrix: {matrix_name}")
    for ld, coverage in ld_space.items():
        zero_try = sum(1 for coverage in ld_space[ld] if coverage == 0)


# Dictionary containing the lengths of the matrices
matrix_lengths = {
    "Porebski_hb": len(hb_pb_matrix),
    "Porebski_lb": len(lb_pb_matrix),
    "Tz_25k_hb": len(hb_tz_matrix),
    "Tz_25k_lb": len(lb_tz_matrix)
     
}

# Compute zero counts and normalized zero counts
zero_counts_df = compute_zero_counts(ld_values, ld_space_dict, matrix_lengths)

# Flatten coverage data for plotting
coverage_data = flatten_coverage_data(ld_values, ld_space_dict)

# Display the zero counts DataFrame
print(zero_counts_df)


# Ensure proper styles for plotting
sns.set(style="whitegrid")


fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

# Top Plot: Barplot for normalized zero counts
sns.barplot(
    data=zero_counts_df, x='LD', y='Normalized_Zero_Count', hue='Matrix',
    ax=axes[0], dodge=True, palette='muted'
)
axes[0].set_title('Fraction of Zero LD Space Coverage sequences for Porebski and TZ Matrices')
axes[0].set_ylabel('Fraction of Zero on whole dataset')
axes[0].legend(title='Matrix')

# Bottom Plot: Boxplot for non-zero LD coverage
sns.boxplot(
    data=coverage_data, x='LD', y='Coverage', hue='Matrix',
    ax=axes[1], dodge=True, palette='muted'
)
axes[1].set_yscale('log')
axes[1].set_title('Single sequence LD Space Coverage (Non-Zero) for Porebski and TZ Matrices')
axes[1].set_xlabel('Levenshtein Distance (LD)')
axes[1].set_ylabel('Coverage (Log Scale)')
axes[1].legend(title='Matrix')

plt.tight_layout()
# plt.savefig('LD_coverage_boxplot_COMBINED.png')
plt.savefig('LD_coverage_boxplot_COMBINED_SAMPLE25k_HB_LB.png')
plt.show()

#compute sample LD coverage
df_sample_filt = filter_data(tz_metadata,
                                sample_size= 10000,
                                junction_aa_col='cdr3_aa',
                                affinity_col='binding_label')




#compute position entropy
import numpy as np

def compute_position_entropy_numpy(sequences):
    """
    Compute per-position Shannon entropy for a list of amino acid sequences using NumPy.
    
    :param sequences: List of strings, each string representing an amino acid sequence.
                      All sequences should have the same length.
    :return: 1D NumPy array of floats representing the entropy at each position.
    """
    if not sequences:
        raise ValueError("The sequence list is empty.")
        
    seq_length = len(sequences[0])
    num_sequences = len(sequences)
    
    # Check if all sequences have the same length
    for seq in sequences:
        if len(seq) != seq_length:
            raise ValueError("All sequences must have the same length.")
    
    # Convert list of strings into a 2D character array
    # shape = (num_sequences, seq_length)
    arr = np.array([list(seq) for seq in sequences])
    
    # We want frequencies for each position
    # One way is to iterate over each column (position), but let's do it more systematically
    position_entropies = np.zeros(seq_length, dtype=float)
    
    # Convert each amino acid to a numeric label to compute counts quickly
    # We can map each unique amino acid to an integer using np.unique with 'return_inverse'
    unique_aas, inverse = np.unique(arr, return_inverse=True)
    # 'inverse' is a 1D array of labels for each character in arr, flattened in row-major order
    print("Unique amino acids:", unique_aas)
    # Reshape 'inverse' to the same shape as arr
    label_matrix = inverse.reshape(arr.shape)
    
    # For each position (column), we want to count occurrences of each label
    for pos in range(seq_length):
        # Extract the column of labels
        column_labels = label_matrix[:, pos]
        
        # Count how many times each label occurs
        counts = np.bincount(column_labels)
        
        # Convert to probabilities
        p = counts / num_sequences
        
        # Filter out zero entries to avoid issues in log2
        p_nonzero = p[p > 0]
        
        # Compute entropy
        entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
        position_entropies[pos] = entropy
    
    return position_entropies


# Example usage
tz_metadata= pd.read_csv('/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv', sep='\t')
porebski_metadata= pd.read_csv('/doctorai/userdata/airr_atlas/data/sequences/bcr/porebski_metadata.csv')

tz_sequences = tz_metadata['cdr3_aa'].tolist()
porebski_sequences = porebski_metadata['cdr3'].tolist()

tz_metadata['binding_label'].value_counts()

tz_sequences[:5]


# Get TZ sequences per binding label
PB_sequences_hb = porebski_metadata[porebski_metadata['binding_label'] == 'hb']['cdr3'].tolist()
PB_sequences_lb = porebski_metadata[porebski_metadata['binding_label'] == 'lb']['cdr3'].tolist()

TZ_sequences_hb = tz_metadata[tz_metadata['binding_label'] == 'hb']['cdr3_aa'].tolist()
TZ_sequences_lb = tz_metadata[tz_metadata['binding_label'] == 'lb']['cdr3_aa'].tolist()

# Compute entropies for each binding label
entropies_PB_hb = compute_position_entropy_numpy(TZ_sequences_hb)
entropies_PB_lb = compute_position_entropy_numpy(TZ_sequences_lb)

print("Position Entropies for HB (length 15) [NumPy]:", entropies_PB_hb)
print("Position Entropies for LB (length 15) [NumPy]:", entropies_PB_lb)

# Calculate the sum of entropies for each binding label
sum_entropy_PB_hb = np.sum(entropies_PB_hb)
sum_entropy_PB_lb = np.sum(entropies_PB_lb)

print(f"Sum of Entropies for TZ HB dataset: {sum_entropy_PB_hb}")
print(f"Sum of Entropies for TZ LB dataset: {sum_entropy_PB_lb}")

# Plot positional entropy for each binding label
def plot_position_entropy_per_label(entropies_hb, entropies_lb):
    positions = np.arange(1, len(entropies_hb) + 1)
    
    plt.figure(figsize=(12, 6))
    
    width = 0.35  # Width of the bars
    
    plt.bar(positions - width/2, entropies_hb, width=width, label='HB', color='green', alpha=0.7)
    plt.bar(positions + width/2, entropies_lb, width=width, label='LB', color='red', alpha=0.7)
    
    plt.xlabel("Position")
    plt.ylabel("Entropy (bits)")
    plt.title("Positional Entropy Comparison per Binding Label")
    plt.xticks(np.arange(1, len(entropies_hb) + 1))
    plt.legend()
    plt.savefig('Positional_Entropy_Comparison_per_Label.png')
    plt.show()

plot_position_entropy_per_label(entropies_PB_hb, entropies_PB_lb)

entropies_Tz = compute_position_entropy_numpy(tz_sequences)
entropies_POREBSKI = compute_position_entropy_numpy(porebski_sequences)

print("Position Entropies (length 15) [NumPy]:", entropies_Tz)
print("Position Entropies (length 21) [NumPy]:", entropies_POREBSKI)
# Calculate the sum of entropies for both datasets
sum_entropy_Tz = np.sum(entropies_Tz)
sum_entropy_POREBSKI = np.sum(entropies_POREBSKI)

print(f"Sum of Entropies for TZ dataset: {sum_entropy_Tz}")
print(f"Sum of Entropies for POREBSKI dataset: {sum_entropy_POREBSKI}")


import matplotlib.pyplot as plt

def plot_position_entropy(entropies_Tz, entropies_POREBSKI):
    """
    Plot the positional entropy for both datasets using matplotlib.
    
    :param entropies_Tz: List or array of entropy values for the TZ dataset.
    :param entropies_POREBSKI: List or array of entropy values for the POREBSKI dataset.
    """
    positions_Tz = np.arange(1, len(entropies_Tz) + 1)
    positions_POREBSKI = np.arange(1, len(entropies_POREBSKI) + 1)
    
    plt.figure(figsize=(12, 6))
    
    width = 0.35  # Width of the bars
    
    plt.bar(positions_Tz - width/2, entropies_Tz, width=width, label='TZ', color='blue', alpha=0.7)
    plt.bar(positions_POREBSKI + width/2, entropies_POREBSKI, width=width, label='POREBSKI', color='orange', alpha=0.7)
    
    plt.xlabel("Position")
    plt.ylabel("Entropy (bits)")
    plt.title("Positional Entropy Comparison")
    plt.xticks(np.arange(1, max(len(entropies_Tz), len(entropies_POREBSKI)) + 1))
    plt.legend()
    plt.savefig('Positional_Entropy_Comparison.png')
    plt.show()

plot_position_entropy(entropies_Tz, entropies_POREBSKI)
