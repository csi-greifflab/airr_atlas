
import numpy as np
from scipy.sparse import coo_matrix
import sys
import time
import os
import glob
import re
from scipy.sparse.linalg import eigsh
from scipy.sparse import load_npz
from scipy.sparse import save_npz
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import laplacian

vicinity_100k_4000knn=vicinity_analysis_instance


def compute_adjacency_matrices(vicinity_instance, density_thresholds):
    """
    Computes sparse adjacency matrices for each density threshold.

    Parameters:
    -----------
    vicinity_instance : object
        An object that contains the attributes:
            - NN_id: a 2D numpy array where the first column contains point IDs,
                     and subsequent columns contain the IDs of neighboring points.
            - NN_dist: a 2D numpy array of the same shape as NN_id where each row
                       contains the distances from the point (first column) to its neighbors.
    density_thresholds : list or array-like
        A list of threshold values. For each threshold, only the neighbor distances
        less than or equal to the threshold (excluding self-distance) are considered.

    Returns:
    --------
    adj_mat_list : list
        A list of sparse adjacency matrices (in CSR format), one for each threshold.
    """
    # Precompute row identifiers from the first column of NN_id.
    row_ids_all = vicinity_instance.NN_id[:, 0]
    n = len(row_ids_all)
    
    # Create a mapping array to quickly map point IDs to their row indices.
    # We assume point IDs are non-negative integers.
    unique_ids = np.unique(row_ids_all)
    mapping = np.zeros(unique_ids.max() + 1, dtype=int)
    for idx, row_id in enumerate(row_ids_all):
        mapping[row_id] = idx

    # Prepare a list to hold the resulting adjacency matrices.
    adj_mat_list = [None] * len(density_thresholds)
    start = time.time()
    # For each density threshold, compute the adjacency matrix.
    for i_density, thr_density in tqdm(enumerate(density_thresholds), total=len(density_thresholds)):
        # Create a boolean mask where NN_dist <= threshold.
        mask = vicinity_instance.NN_dist <= thr_density
        # Exclude self-connections: assume first column corresponds to self.
        mask[:, 0] = False

        # Create an array with repeated row indices for each neighbor.
        repeated_rows = np.repeat(np.arange(n), vicinity_instance.NN_dist.shape[1])
        
        # Flatten the NN_id matrix to align with the flattened mask.
        flattened_neighbor_ids = vicinity_instance.NN_id.flatten()
        mask_flat = mask.flatten()
        
        # Select valid row indices and neighbor IDs using the mask.
        valid_row_indices = repeated_rows[mask_flat]
        valid_neighbor_ids = flattened_neighbor_ids[mask_flat]
        
        # Map valid neighbor IDs to row indices using the mapping array.
        valid_neighbor_indices = mapping[valid_neighbor_ids]
        
        # To create a symmetric adjacency matrix, add both (i, j) and (j, i).
        all_rows = np.concatenate([valid_row_indices, valid_neighbor_indices])
        all_cols = np.concatenate([valid_neighbor_indices, valid_row_indices])
        data = np.ones(all_rows.shape[0], dtype=int)
        
        # Build the sparse matrix in COO format and convert to CSR.
        adjacency_matrix = coo_matrix((data, (all_rows, all_cols)), shape=(n, n)).tocsr()
        adj_mat_list[i_density] = adjacency_matrix
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")
    return adj_mat_list


def compute_adjacency_matrices_NONSYM(vicinity_instance, density_thresholds):
    """
    Computes sparse adjacency matrices for each density threshold.

    Parameters:
    -----------
    vicinity_instance : object
        An object that contains the attributes:
            - NN_id: a 2D numpy array where the first column contains point IDs,
                     and subsequent columns contain the IDs of neighboring points.
            - NN_dist: a 2D numpy array of the same shape as NN_id where each row
                       contains the distances from the point (first column) to its neighbors.
    density_thresholds : list or array-like
        A list of threshold values. For each threshold, only the neighbor distances
        less than or equal to the threshold (excluding self-distance) are considered.

    Returns:
    --------
    adj_mat_list : list
        A list of sparse adjacency matrices (in CSR format), one for each threshold.
    """
    # Precompute row identifiers from the first column of NN_id.
    row_ids_all = vicinity_instance.NN_id[:, 0]
    n = len(row_ids_all)
    
    # Create a mapping array to quickly map point IDs to their row indices.
    # We assume point IDs are non-negative integers.
    unique_ids = np.unique(row_ids_all)
    mapping = np.zeros(unique_ids.max() + 1, dtype=int)
    for idx, row_id in enumerate(row_ids_all):
        mapping[row_id] = idx

    # Prepare a list to hold the resulting adjacency matrices.
    adj_mat_list = [None] * len(density_thresholds)
    start = time.time()
    # For each density threshold, compute the adjacency matrix.
    for i_density, thr_density in enumerate(density_thresholds):
        # Create a boolean mask where NN_dist <= threshold.
        mask = vicinity_instance.NN_dist <= thr_density
        # Exclude self-connections: assume first column corresponds to self.
        mask[:, 0] = False

        # Create an array with repeated row indices for each neighbor.
        repeated_rows = np.repeat(np.arange(n), vicinity_instance.NN_dist.shape[1])
        
        # Flatten the NN_id matrix to align with the flattened mask.
        flattened_neighbor_ids = vicinity_instance.NN_id.flatten()
        mask_flat = mask.flatten()
        
        # Select valid row indices and neighbor IDs using the mask.
        valid_row_indices = repeated_rows[mask_flat]
        valid_neighbor_ids = flattened_neighbor_ids[mask_flat]
        
        # Map valid neighbor IDs to row indices using the mapping array.
        valid_neighbor_indices = mapping[valid_neighbor_ids]
        
        # To create a symmetric adjacency matrix, add both (i, j) and (j, i).
        # Only use the outgoing edges
        all_rows = valid_row_indices
        all_cols = valid_neighbor_indices
        data = np.ones(all_rows.shape[0], dtype=int)

        
        # Build the sparse matrix in COO format and convert to CSR.
        adjacency_matrix = coo_matrix((data, (all_rows, all_cols)), shape=(n, n)).tocsr()
        adj_mat_list[i_density] = adjacency_matrix
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")
    return adj_mat_list


# Example usage:
density_thresholds = [0.001,0.002, 0.003, 0.004,0.005, 0.006]
adjacency_matrices = compute_adjacency_matrices(vicinity_analysis_instance, density_thresholds)
adjacency_matrices_NONSYM = compute_adjacency_matrices_NONSYM(vicinity_analysis_instance, density_thresholds)
vicinity_analysis_instance.NN_dist.shape[1]

adjacency_matrices=adjacency_matrices_NONSYM


neighbors_count = adjacency_matrices[1].getnnz(axis=1)
# save_npz('/doctorai/niccoloc/airr_atlas/scripts/Vicinity_code/adjacency_matrix.npz', adjacency_matrices[1])

# Read the saved adjacency matrix file
adjacency_matrix = load_npz('/doctorai/niccoloc/airr_atlas/scripts/Vicinity_code/adjacency_matrix.npz')
print("Loaded adjacency matrix shape:", adjacency_matrix.shape)
print("Number of non-zero entries:", adjacency_matrix.nnz)


# Compute the Laplacian matrix
laplacian_matrix = laplacian(adjacency_matrices[1], normed=False)

# Count the number of neighbours (non-zero entries) for each row in adjacency_matrices[1]
neighbors_count = adjacency_matrices[1].getnnz(axis=1)
print("Number of neighbours for each row in adjacency_matrices[1]:")
print(neighbors_count)
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import laplacian




import matplotlib
matplotlib.use('Agg')


plt.figure(figsize=(30, 30))
plt.spy(adjacency_matrices[1][:20000,:20000],marker="s", markersize=0.01, rasterized=True, alpha=1)
plt.title("Adjacency Matrix 1")
plt.xlabel("Index")
plt.ylabel("Index")
plt.tight_layout()
plt.savefig("/doctorai/niccoloc/airr_atlas/scripts/Vicinity_code/adjacency_matrix_3.png", dpi = 300)
plt.clf()


n_matrices = len(adjacency_matrices)

# Precompute all required data using list comprehensions

lap_diags = [laplacian(mat, normed=False).diagonal() for mat in adjacency_matrices]
neighbors_counts = [mat.getnnz(axis=1) for mat in adjacency_matrices]

n_bin=100
fig, axs = plt.subplots(n_matrices, 2, figsize=(18, 6 * n_matrices))
if n_matrices == 1:
    axs = np.array([axs])

for i in range(n_matrices):
    # Adjacency heatmap
    # dense_adjs=adjacency_matrices[i].toarray()
    # im0 = axs[i, 0].imshow(dense_adjs, cmap='viridis', interpolation='none')
    # axs[i, 0].set_title(f'Adjacency Matrix cosine:{density_thresholds[i]}')
    # axs[i, 0].set_xlabel('Index')
    # axs[i, 0].set_ylabel('Index')
    # fig.colorbar(im0, ax=axs[i, 0], fraction=0.046, pad=0.04)
    # Instead of converting to dense, use plt.spy to visualize the sparsity pattern
    # # axs[i, 0].spy(adjacency_matrices[i] ,marker="s", markersize=0.01, rasterized=True, alpha=0.3)
    # # axs[i, 0].set_title(f'Adjacency Matrix {i+1}')
    # # axs[i, 0].set_xlabel('Index')
    # # axs[i, 0].set_ylabel('Index')
    # if i >= 1:
    #     break
    # Laplacian diagonal histogram
    axs[i, 0].hist(lap_diags[i], bins=n_bin, color='blue', alpha=0.7)
    count_zero = np.sum(lap_diags[i] == 0)
    axs[i, 0].text(0.95, 0.95, f"0 instances: {count_zero}", transform=axs[i, 0].transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right')
    axs[i, 0].set_title(f'Laplacian Diagonal Histogram cosine:{density_thresholds[i]}')
    axs[i, 0].set_xlabel('Diagonal Value')
    axs[i, 0].set_ylabel('Frequency')
    
    # Neighbors count histogram
    axs[i, 1].hist(neighbors_counts[i], bins=n_bin, color='green', alpha=0.7)
    count_zero_neighbors = np.sum(neighbors_counts[i] == 0)
    axs[i, 1].text(0.95, 0.95, f"0 instances: {count_zero_neighbors}", transform=axs[i, 1].transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right')
    axs[i, 1].set_title(f'Neighbors Count Histogram cosine:{density_thresholds[i]}')
    axs[i, 1].set_xlabel('Number of Neighbors')
    axs[i, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("/doctorai/niccoloc/airr_atlas/scripts/Vicinity_code/composite_plot_efficient_NONSYM_4000_10keach.png")
plt.clf()



vicinity_analysis_instance.NN_dist
within_threshold_indices=np.where(vicinity_analysis_instance.NN_dist[0] <= thr_density)[0] [1:]
KNN_ids=vicinity_analysis_instance.NN_id[0, within_threshold_indices]
KNN_ids


# test for Multiple vicinity


vicinity_paths = "/doctorai/niccoloc/Vicinity_results_100k_Density"

x1= load_npz('/AttentionMat_ab2_cdr3_only_layer_1/adjacency_matrix_R0001_AttentionMat_ab2_cdr3_only_layer_1.npz')
x2= load_npz('/doctorai/niccoloc/Vicinity_results_100k_Density/Pooled_ab2_cdr3_only_layer_12/adjacency_matrix_R0005_Pooled_ab2_cdr3_only_layer_12.npz')

#test for density
import time
vicinity_analysis_instance.NN_id
from scipy.sparse import lil_matrix
import numpy as np
from scipy.sparse.csgraph import laplacian
density_thresholds=[0.005,0.008,0.01]
adj_mat_list=[None]*len(density_thresholds)
start = time.time()
for i_density, thr_density in enumerate(density_thresholds):
        print(f'Computing Density at thr {thr_density}...')
        row_ids = vicinity_analysis_instance.NN_id[:, 0]  # get all the row ids
        n = len(row_ids)
        # Create a sparse matrix in LIL format (efficient for row-based construction)
        adjacency_matrix = lil_matrix((n, n), dtype=int)
        row_index_map = {row_id: idx for idx, row_id in enumerate(row_ids)}
        for i in range(vicinity_analysis_instance.NN_dist.shape[0]):  # loop through all the points
            row_ids = vicinity_analysis_instance.NN_id[i][0]  # get the point itself
            current_row_index = row_index_map[row_ids]
            if i == 20:
                print(f"this is the current_row_index {current_row_index}, and this is the row_ids {row_ids}")
            neighbors_indices = vicinity_analysis_instance.NN_id[i][1:]  # exclude the point itself
            within_threshold_indices = np.where(vicinity_analysis_instance.NN_dist[i] <= thr_density)[0][1:]  # Exclude the point itself
            KNN_ids=vicinity_analysis_instance.NN_id[i, within_threshold_indices]
            adjacency_matrix[current_row_index, KNN_ids] = 1
            adjacency_matrix[KNN_ids, current_row_index] = 1 
            adjacency_matrix[KNN_ids, KNN_ids] = 0
        adj_mat_list[i_density]  = adjacency_matrix
        print(f'Adjacency matrix at thr {thr_density} done')
end = time.time()
print(f"Time elapsed: {end - start} seconds")        
import matplotlib.pyplot as plt





#trying to estimante lambda connectivity

import os
import glob
import re
from scipy.sparse.linalg import eigsh

list_vicinity_paths=["/doctorai/niccoloc/Vicinity_results_100k_Densit.*"]


# Define the base directory to search in
base_dir = "/doctorai/niccoloc"


# Find all directories in base_dir that start with "Vicinity_results_100k_Density"
vicinity_dirs = [ 
    d for d in glob.glob(os.path.join(base_dir, "Vicinity_results_100k_Densit*"))
    if os.path.isdir(d)
]


vicinity_dirs_porebski = [ 
    d for d in glob.glob(os.path.join(base_dir, "Vicinity_porebski_Densit*"))
    if os.path.isdir(d)
]

knn_density_thr=[10,50,100,500,1000]


vicinity_dirs= vicinity_dirs + vicinity_dirs_porebski

# matrix_data_tz = matrix_data
# Read files starting with "adjacency_matrix" from each vicinity directory and add them to a list
adjacency_info_dict = {}
adjacency_matrix_files = []
for directory in vicinity_dirs:
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        #select only the layer 0
        if os.path.isdir(subdir_path) and not "sample" in subdir_path:
        # if os.path.isdir(subdir_path):
            pattern = os.path.join(subdir_path, "adjacency_matrix*.npz")
            print(pattern)
            files = glob.glob(pattern)
            print(files)
            adjacency_matrix_files.extend(files)



            
            # filename = '/doctorai/niccoloc/Vicinity_results_100k_Density_sample25000/Pooled_ab2_cdr3_only_embeddings_sample_25000_layer_0/adjacency_matrix_NN_R00458_Pooled_ab2_cdr3_only_embeddings_sample_25000_layer_0.npz'

# for filename in adjacency_matrix_files:
# Extract the radius: value after 'NN_R'
filename = adjacency_matrix_files[0]

for filename in adjacency_matrix_files:
    # For testing, override filename with a specific file
    # filename = adjacency_matrix_files[12]
    print("Filename:", filename)
    # This pattern captures:
    # - 'prefix' as NN, lin, or matrix
    # - 'number' as the number immediately following the prefix
    # - 'radius' as the number following '_R' (if present)
    # - 'after' as any additional characters up to '.npz'
    pattern = r'(?P<prefix>NN|lin|matrix)(?P<number>\d+)(?:_R(?P<radius>\d+))?(?:_(?P<after>[^.]+))?\.npz$'
    match = re.search(pattern, filename)
    if match:
        prefix = match.group('prefix')
        number = match.group('number')
        radius = match.group('radius')
        after = match.group('after')
        if prefix == 'NN':
            prefix_type = 'NN'
        elif prefix == 'lin':
            prefix_type = 'lin'
        elif prefix == 'matrix':
            prefix_type = 'R'
    else:
        number = 6
        radius = None
        after = None
        prefix_type = "fixed"

    print("Prefix type:", prefix_type)
    print("Number:", number)
    print("Radius:", radius)
    print("Extra after prefix:", after)
    # Detect dataset from filename: use its parent directory name as the dataset identifier.
    if "porebski" in filename:
        dataset = "porbeski"
    else:
        dataset = "tz"

    # Extract the layer: from 'layer_' followed by digits
    layer_match = re.search(r'layer_(\d+)', filename)
    layer = layer_match.group(1) if layer_match else None

    # extract pooling method and detect if it is either "pooled", "unpooled" or "attentionmat"
    pooling_match = re.search(r'(?i)(Pooled|Unpooled|AttentionMat)', filename)
    pooling = pooling_match.group(1).lower() if pooling_match else None

    # Extract the sample size: from 'sample' followed by digits
    sample_match = re.search(r'sample_(\d+)', filename)
    sample_size = sample_match.group(1) if sample_match else None

    # Load the adjacency matrix
    # adjacency_matrix = load_npz(filename)
    # #compute the laplacian matrix
    # laplacian_matrix = laplacian(adjacency_matrix, normed=False).astype(np.float32, copy=False)
    # neighbors_counts = adjacency_matrix.getnnz(axis=1)
    # connectivity = compute_connectivity(laplacian_matrix)
    # print("Connectivity:", connectivity)
    # effective_densities = effective_density(neighbors_counts, connectivity)
    # print("Connectivity:", connectivity)
    # print("Effective densities:", effective_densities)
    # Store the extracted details using the filename as the key.


    adjacency_info_dict[filename] = {
    "radius": radius,
    "prefix": prefix_type,
    "thr": number,
    "layer": layer,
    "dataset" : dataset,
    # "matrix": adjacency_matrix,
    "sample_size": sample_size,
    "layer": layer,
    # "laplacian": laplacian_matrix,
    # "neighbors_counts": neighbors_counts,
    "filename": filename,
    "pooling": pooling
    # "matrix": adjacency_matrix,
    # "connectivity": connectivity,
    # "effective_densities": effective_densities
    }
    print("Found adjacency matrix files:")



import pandas as pd

# Create a DataFrame from adjacency_info_dict using the filenames as the index.
df_adjacency = pd.DataFrame.from_dict(adjacency_info_dict, orient='index')

filtered_df = df_adjacency[~df_adjacency['thr'].isin(['500', '5', '6', '1000'])]
filtered_df = df_adjacency[df_adjacency['thr'].isin(['0', '1', '10',  '50'])]


# Get the filename given a specific prefix and thr
desired_prefix = "NN"  # change as needed
desired_thr = "10"    # change as needed (as a string)

filtered = df_adjacency[(df_adjacency["prefix"] == desired_prefix) & (df_adjacency["thr"] == desired_thr) & (df_adjacency['sample_size'].isnull())]
if not filtered.empty:
    fn1 = filtered.index[0]
    print(f"Matching filename found: {filtered}")
else:
    fn1 = None
    print("No matching filename found for the given prefix and thr.")


print(filtered_df)


# Load matrices from each filename in the filtered DataFrame into a dictionary


matrix_data = { }
for filename in filtered_df['filename']:
    try:
        matrix = load_npz(filename)
        print(f"Loaded matrix from {filename} with shape {matrix.shape}")
    except Exception as e:
        print(f"Failed to load matrix from {filename}: {e}")
    laplacian_matrix = laplacian(matrix, normed=False).astype(np.float32, copy=False)
    diagonal = laplacian_matrix.diagonal()

    neighbors_counts = matrix.getnnz(axis=1)
    matrix_data[filename] = {
        "matrix": matrix,
        "laplacian": laplacian_matrix,
        "diagonal": diagonal,
        "neighbors_counts": neighbors_counts
    }
    print(f"Computed Laplacian matrix and neighbors counts for {filename}")
    break

color_map = {
    0: 'red',
    25000: 'blue',
    10000: 'green',
    5000: 'orange',
    1000: 'purple',
    500: 'yellow'
}

    # Create a single plot with multiple histogram lines for neighbor counts
# Identify unique pooling categories across all matrices
pooling_categories = ['attentionmat', 'pooled', 'unpooled']
# Determine the unique pooling categories and thresholds (excluding thr==6)
unique_poolings = pooling_categories  # e.g. ['attentionmat', 'pooled', 'unpooled']
# Gather unique thr values from the dataframe (skip thr==6)
unique_thrs = sorted({metadata["thr"] for metadata in filtered_df.to_dict('index').values() if metadata["thr"] != 6},key=int)

n_rows = len(unique_poolings)
n_cols = len(unique_thrs)

# Create a 2D grid of subplots: rows for pooling categories, columns for density threshold (thr)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows), sharex=True, sharey=True)
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = np.array([axes])
elif n_cols == 1:
    axes = np.array([[ax] for ax in axes])

# Create mapping dictionaries for pooling and thr to subplot grid indices
pooling_to_row = {pool: idx for idx, pool in enumerate(unique_poolings)}
thr_to_col = {thr: idx for idx, thr in enumerate(unique_thrs)}

linestyles = {'NN': '-', 'lin': '--', 'R': ':'}

# Loop over the data and plot each histogram in its corresponding subplot cell
for fname, info in matrix_data.items():
    metadata = df_adjacency.loc[fname]
    pooling = metadata["pooling"]
    prefix = metadata["prefix"]
    thr = metadata["thr"]
    radius = metadata["radius"]
    sample_size = metadata["sample_size"]
    counts = info["neighbors_counts"]
    dataset = metadata["dataset"]
    if thr in [1000 ,3, 4, 5,500, 6]:
        continue

    # Get the grid cell based on pooling category and threshold value
    row = pooling_to_row[pooling]
    col = thr_to_col[thr]
    ax = axes[row, col]

    print(f"Plotting histogram for {prefix}{thr}_R{radius}_S{sample_size} (pooling: {pooling})")

    # Compute histogram data manually
    hist, bins = np.histogram(counts, bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if sample_size is None:
        sample_size = 100000
    sample_size_int = int(sample_size) if int(sample_size) > 0 else 1
    hist = hist / sample_size_int
    if sample_size == 100000:
        sample_size = "0"
        
    color = color_map.get(int(sample_size), 'black')
    if dataset == "porbeski":
        color = 'magenta'
        print("Porbeski")

    label = f"{prefix}{thr}_R{radius}_S{sample_size}_{dataset}"
    linestyle = linestyles.get(prefix, '-')
    
    ax.plot(bin_centers, hist, label=label, linestyle=linestyle, color=color, alpha=1, linewidth=0.5)
    ax.set_title(f"{pooling} - thr: {prefix} {thr}")
    ax.legend()
    ax.set_xlim(0, 200)
    # ax.set_ylim(0, 1.2)

# Set common labels
for ax in axes[-1, :]:
    ax.set_xlabel("Neighbor Count")
for ax in axes[:, 0]:
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("/doctorai/niccoloc/airr_atlas/scripts/Vicinity_code/neighbors_count_histogram.png")
# plt.savefig("/doctorai/niccoloc/airr_atlas/scripts/Vicinity_code/neighbors_count_histogram_POREBSKI.png")
plt.clf()




stats_dict = {}
# Loop over the data and plot statistics in each corresponding subplot cell
for fname, info in matrix_data.items():
    metadata = df_adjacency.loc[fname]
    pooling = metadata["pooling"]
    prefix = metadata["prefix"]
    thr = metadata["thr"]
    radius = metadata["radius"]
    sample_size = metadata["sample_size"]
    counts = info["neighbors_counts"]
    dataset = metadata["dataset"]
    if thr == '1':
        print(counts)

    if thr == 6:
        continue
    # Compute mean, median and the frequency distribution
    mean_val = np.mean(counts)
    median_val = np.median(counts)
    unique_vals, freq = np.unique(counts, return_counts=True)
    std_mean = np.std(counts)

    
    # Save the results into a dictionary
    stats_dict[fname] = {
        "mean": mean_val,
        "median": median_val,
        "unique_values": unique_vals,
        "frequency": freq,
        "std_mean": std_mean,
        "zero_hist": freq[0],
    }


# Compute and plot statistics for neighbor counts instead of a histogram
# Identify unique pooling categories across all matrices
pooling_categories = ['attentionmat', 'pooled', 'unpooled']
# Determine the unique pooling categories and thresholds (excluding thr==6)
unique_poolings = pooling_categories  # e.g. ['attentionmat', 'pooled', 'unpooled']
# Gather unique thr values from the dataframe (skip thr==6)
unique_thrs = sorted({metadata["thr"] for metadata in filtered_df.to_dict('index').values() if metadata["thr"] != 6}, key =int)
n_rows = len(unique_poolings)
n_cols = len(unique_thrs)

# Create a 2D grid of subplots: rows for pooling categories, columns for density threshold (thr)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows), sharex=True, sharey=True)
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = np.array([axes])
elif n_cols == 1:
    axes = np.array([[ax] for ax in axes])

# Create mapping dictionaries for pooling and thr to subplot grid indices
pooling_to_row = {pool: idx for idx, pool in enumerate(unique_poolings)}
thr_to_col = {thr: idx for idx, thr in enumerate(unique_thrs)}

linestyles = {'NN': '-', 'lin': '--', 'R': ':'}

for fname, info in matrix_data.items():
    metadata = df_adjacency.loc[fname]
    pooling = metadata["pooling"]
    prefix = metadata["prefix"]
    thr = metadata["thr"]
    radius = metadata["radius"]
    sample_size = metadata["sample_size"]
    counts = info["neighbors_counts"]
    dataset = metadata["dataset"]

    if thr == 6:
        continue

    mean_val = stats_dict[fname]["mean"]
    median_val = stats_dict[fname]["median"]
    unique_vals = stats_dict[fname]["unique_values"]
    freq = stats_dict[fname]["frequency"]
    std_mean = stats_dict[fname]["std_mean"]
    zero_count = stats_dict[fname]["zero_hist"]
    # Get the grid cell based on pooling category and threshold value
    row = pooling_to_row[pooling]
    col = thr_to_col[thr]
    ax = axes[row, col]

    print(f"Plotting stats for {prefix}{thr}_R{radius}_S{sample_size} (pooling: {pooling})")
    if sample_size is None:
        sample_size = 100000
    sample_size_int = int(sample_size) if int(sample_size) > 0 else 1
    freq = freq / sample_size_int
    zero_count = zero_count / sample_size_int
    if sample_size == 100000:
        sample_size = "0"
        

    label = f"{prefix}{thr}_R{radius}_S{sample_size}"
    linestyle = linestyles.get(prefix, '-')
    if dataset == "porbeski":
        colors = 'magenta'
    else:
        colors = color_map.get(int(sample_size), 'black')

    ax.plot(unique_vals, freq, color=colors, alpha=0.6, label="Frequency")

    label_mean = f"{dataset}-Mean: {mean_val:.2f}_std{std_mean:.2f}_S{sample_size}_Z{zero_count}"
    label_median = f"{dataset}-Median: {median_val:.2f}_S{sample_size}_Z{zero_count}"


    # Plot vertical lines for mean and median
    ax.axvline(mean_val, color=colors, linestyle='-', linewidth=1.5, label=label_mean)
    ax.axvline(median_val, color=colors, linestyle=':', linewidth=1.5, label= label_median)
    #print as addtiona il info the frequency of zeros
    print(f"freq of zeros: {freq[0]}")
    count_zero =  freq[0]


    ax.set_title(f"{pooling} - thr: {prefix} {thr}")
    ax.set_xlabel("Neighbor Count")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.8)
    ax.legend()

plt.tight_layout()
plt.savefig("/doctorai/niccoloc/airr_atlas/scripts/Vicinity_code/neighbors_count_statistics.png")
# plt.savefig("/doctorai/niccoloc/airr_atlas/scripts/Vicinity_code/neighbors_count_statistics_POREBSKI.png")
plt.clf()




print("Radius:", radius)
print("Sample size:", sample_size)
print("Layer:", layer) 
# Save extracted information into a dictionary.
# Create the dictionary if it doesn't already exist.
if 'adjacency_info_dict' not in globals():
    adjacency_info_dict = {}

adjacency_matrix=load_npz(filename)   
# Store the extracted details using the filename as the key.
adjacency_info_dict[filename] = {
    "radius": radius,
    "sample_size": sample_size,
    "layer": layer,
    "matrix": adjacency_matrix
}
 
print("Found adjacency matrix files:")
for file in adjacency_matrix_files:
    print(file)




def compute_connectivity(L):
    """
    Compute the effective density for each node in the graph.
        L : scipy.sparse.csr_matrix
            The Laplacian matrix of the graph.
    """
    # Compute the two smallest eigenvalues; use which='SM' (smallest magnitude)
    # For symmetric matrices like the Laplacian, eigsh is efficient.
    eigenvalues, _ = eigsh(L, k=2, which='SM')
    
    # Sort the computed eigenvalues to ensure proper ordering
    eigenvalues_sorted = np.sort(np.real(eigenvalues))
    
    # The Fiedler value is the second smallest eigenvalue
    fiedler_value = eigenvalues_sorted[1]
    return fiedler_value

 


def effective_density(degrees, connectivity):
    """
    Compute the effective density for each node in the graph.
        adjacency_matrix : scipy.sparse.csr_matrix
            The adjacency matrix of the graph.
    """
    fiedler_value =connectivity
    if fiedler_value == 0:
        effective_density = 0
    else:
        effective_density = degrees / fiedler_value
    
    return effective_density


import time
import pandas as pd
import os
import csv

start = time.time()
# eigenvalues, _ = eigsh(laplacian_matrix.tocsr(), k=50, which='SM')
fn1= '/doctorai/niccoloc/Vicinity_results_100k_Density/Pooled_ab2_cdr3_only_layer_0/adjacency_matrix_NN10_R00179_Pooled_ab2_cdr3_only_layer_0.npz'

laplacian_matrix = matrix_data[fn1]["laplacian"]
 

eigenvalues, _ = eigsh(laplacian_matrix, k=50, sigma=0, which='LM', tol=1e-4, maxiter=5000)
end = time.time()
print(f"Time elapsed: {end - start} seconds")
print("Eigenvalues:", eigenvalues)







# Subsample the matrix: select a random subset of rows/columns.
n_rows = laplacian_matrix.shape[0]
subsample_size = min(1000, n_rows)  # Adjust the subsample size as needed.
np.random.seed(42)  # For reproducibility.
sample_indices = np.sort(np.random.choice(n_rows, subsample_size, replace=False))

# Create the subsampled Laplacian matrix.

subsampled_matrix = laplacian_matrix[sample_indices, :][:, sample_indices]

# Compute eigenvalues on the subsampled matrix.
# Note: k must be less than the size of the subsampled matrix.
k_eigs = min(50, subsample_size - 1)
eigenvalues, _ = eigsh(subsampled_matrix, k=k_eigs, sigma=0, which='LM', tol=1e-4, maxiter=5000)

end = time.time()
print(f"Time elapsed: {end - start} seconds")
print("Eigenvalues:", eigenvalues)












start = time.time()
X = np.random.random((1000, 1000)) - 0.5
X = np.dot(X, X.T)  # create a symmetric matrix


eigenvalues, _ = eigsh(X, k=100, which='SM')
end = time.time()
print(f"Time elapsed: {end - start} seconds")
print("Eigenvalues:", eigenvalues)

X = np.random.random((1000, 1000)) - 0.5
X = np.dot(X, X.T)
eigenvalues, _ = eigsh(  coo_matrix( X), k=100, which='SM')




X = np.random.random((10000, 10000)) - 0.5
X = np.dot(X, X.T)
eigenvalues, _ = eigsh((X), k=100, which='SM')







# Files with exactly "NN_R" (i.e. no digit between NN and _R) or "lin_R"
pattern_nn = re.compile(r'NN_R')
pattern_lin = re.compile(r'lin_R')

# First method: Using a simple loop
files_to_keep = []
files_to_remove = []
for f in adjacency_matrix_files:
    filename = os.path.basename(f)
    # if filename contains undesired pattern, mark for removal
    if pattern_nn.search(filename) or pattern_lin.search(filename):
        files_to_remove.append(f)
    else:
        files_to_keep.append(f)

# Optionally, remove the undesired files from the filesystem:
# for f in files_to_remove:
#     if os.path.exists(f):
#         os.remove(f)
#         print(f"Removed: {f}")
#     else:
#         print(f"File not found: {f}")

# Print the lists to verify
print("Files to keep:")
for f in files_to_keep:
    print(f)

print("\nFiles to remove:")
for f in files_to_remove:
    print(f)

# -----------------------------------------------
# Alternative method: List comprehensions for filtering
files_to_remove_alt = [f for f in file_paths if pattern_nn.search(os.path.basename(f)) or pattern_lin.search(os.path.basename(f))]
files_to_keep_alt = [f for f in file_paths if f not in files_to_remove_alt]

print("\n(Alternative) Files to keep:")
for f in files_to_keep_alt:
    print(f)

print("\n(Alternative) Files to remove:")
for f in files_to_remove_alt:
    print(f)




for filename in filtered_df['filename']:

    neighbors_counts= matrix_data[filename]["neighbors_counts"]
    diagonal = matrix_data[filename]["diagonal"]


    # Write neighbor count and diagonal values for each sequence (row) to a CSV file
    folder = os.path.dirname(filename)
    base =   os.path.basename(filename).replace('.npz', '')
    out_filename = os.path.join(folder, f"neighbors_diag_{base}.csv")

    df = pd.DataFrame({
        "row": np.arange(len(neighbors_counts)),
        "Neighbors_Count": neighbors_counts,
        "Diagonal_Value": diagonal
    })
    df.to_csv(out_filename, index=False)
    print(f"Written neighbor counts and diagonal values to {out_filename}")



import pandas as pd

LD100k_raw= pd.read_csv("/doctorai/niccoloc/Vicinity_results_sample_test/LD_score_100000/d_whole_LD_stats_LD_score_100000_97.049k.csv"  )

print(type(LD100k_raw.iloc[0, 2]))

import pandas as pd

# Assume df is your original DataFrame (e.g., LD100k_raw)
rows_list = []
global_index = 0

for _, row in LD100k_raw.iterrows():
    threshold = row['LD_thr']
    original_id = row['Unnamed: 0']
    # Iterate through each affinity in the dictionary column
    for affinity, percentages in row['vicinity_by_class'].items():
        for perc in percentages:
            rows_list.append({
                'V1': global_index,
                'ID': original_id,
                'Affinity': affinity,
                'Threshold': threshold,
                'Percentage': perc
            })
            global_index += 1

new_df = pd.DataFrame(rows_list)
print(new_df.head())


import ast
import pandas as pd

def safe_literal_eval(x):
    # If x is not a string, assume it's already converted.
    if not isinstance(x, str):
        return x
    # Replace problematic tokens like 'nan' with a valid literal, e.g., 'None'
    x = x.replace('nan', 'None')
    try:
        return ast.literal_eval(x)
    except Exception as e:
        print("Error evaluating:", x, "->", e)
        return None

# Apply the safe conversion to the 'vicinity' column
LD100k_raw['vicinity'] = LD100k_raw['vicinity'].apply(safe_literal_eval)

# Optionally, you can do the same for other columns that contain nested data:
# LD100k_raw['vicinity_by_class'] = LD100k_raw['vicinity_by_class'].apply(safe_literal_eval)





import ast
import pandas as pd

# Define the safe conversion function
def safe_literal_eval(x):
    # If x is not a string, assume it's already converted.
    if not isinstance(x, str):
        return x
    # Replace tokens like 'nan' with a valid literal if needed.
    x = x.replace('nan', 'None')
    try:
        return ast.literal_eval(x)
    except Exception as e:
        print("Error evaluating:", x, "->", e)
        return None


LD100k_raw= d_res1

# Apply the conversion to the relevant columns
LD100k_raw['vicinity'] = LD100k_raw['vicinity'].apply(safe_literal_eval)
LD100k_raw['num_points'] = LD100k_raw['num_points'].apply(safe_literal_eval)
LD100k_raw['vicinity'] = LD100k_raw['vicinity'].apply(safe_literal_eval)


LD100k_raw['vicinity_by_class'] = LD100k_raw['vicinity_by_class'].apply(safe_literal_eval)
LD100k_raw['num_points_by_class'] = LD100k_raw['num_points_by_class'].apply(safe_literal_eval)


# Verify the conversion worked (for example, check the type of the first element in 'vicinity')
print("Type of first cell in 'vicinity':", type(LD100k_raw.iloc[0]['vicinity']))
print("Type of first cell in 'vicinity_by_class':", type(LD100k_raw.iloc[0]['vicinity_by_class']))

# Now, reformat the DataFrame into a long format.
rows_list = []
global_index = 0  # This will serve as a unique identifier (V1) for each long-format row.
LD100k_raw= d_res1
LD100k_raw.iloc[3,7]

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

# Example usage:
new_df = convert_ld_results_to_long_format(LD100k_raw)
print(new_df.head())





max(sample_id)



# Iterate over each row in the original DataFrame.
for ciao, row in LD100k_raw.iterrows():
    threshold = row['LD_thr']
    # original_id = row['Unnamed: 0']
    print(f"Processing row {ciao} with {row} ")

    # Iterate through each affinity in the vicinity_by_class dictionary.
    for affinity, percentages in row['vicinity_by_class'].items():
        # Retrieve the corresponding list of neighbor counts.
        neighbors = row['num_points_by_class'].get(affinity, [])
        # If there is a length mismatch, warn the user.
        if len(percentages) != len(neighbors):
            print(f"Warning: Length mismatch for affinity '{affinity}' in row {original_id}")
        # Loop over the paired elements.
        for i, perc in enumerate(percentages):
            num_neighbors = neighbors[i] if i < len(neighbors) else None
            rows_list.append({
                'V1': global_index,
                'ID': i,
                'Affinity': affinity,
                'Threshold': threshold,
                'Percentage': perc,
                'Neighbors': num_neighbors
            })
            global_index += 1

# Create the new DataFrame in the desired long format.
new_df = pd.DataFrame(rows_list)
print(new_df.head())

#save the new DataFrame to a CSV file
output_csv_path = "/doctorai/niccoloc/Vicinity_results_sample_test/LD_score_100000/LD_score_100000_long_format.csv"
new_df.to_csv(output_csv_path, index=False)



import pandas as pd

def convert_results_to_long_format(result_row, binding_labels, sample_ids):
    """
    Given a result_row (from results_df) and the corresponding binding_labels and sample_ids,
    convert the row into a long-format DataFrame with the following columns:
       - ID: sample ID
       - Affinity: binding label for that sample
       - Threshold: LD threshold (from the result row)
       - Percentage: affinity-specific percentage (from vicinity_by_class)
       - perc_all: overall percentage (from vicinity)
    """
    overall_perc = result_row['vicinity']  # overall percentages for all samples
    grouped_perc = result_row['vicinity_by_class']  # dict mapping affinity to list of percentages
    threshold = result_row['LD_thr']  # threshold value from this result row

    # We'll use counters to index into each affinity-specific list properly.
    counters = {}
    rows = []
    for sample_id, aff, perc_all in zip(sample_ids, binding_labels, overall_perc):
        # Initialize counter for this affinity if not already present
        if aff not in counters:
            counters[aff] = 0
        index_in_group = counters[aff]
        # Get the affinity-specific percentage from the grouped dictionary
        if aff in grouped_perc and index_in_group < len(grouped_perc[aff]):
            perc_aff = grouped_perc[aff][index_in_group]
        else:
            perc_aff = None  # in case of a mismatch
        counters[aff] += 1

        rows.append({
            'ID': sample_id,
            'Affinity': aff,
            'Threshold': threshold,
            'Percentage': perc_aff,
            'perc_all': perc_all
        })

    return pd.DataFrame(rows)


# --- Example usage ---

# Suppose 'results_df' is your output from prepare_data_for_plotting_LD_MAT
# and you want to convert the first row (for example, LD_thr == 1)
# If you need a different threshold, adjust the index or selection accordingly.
result_row = results_df.iloc[0]

# Assume these were computed earlier in your function:
# binding_labels = matrix_indexes['binding_label'].values
# sample_ids = matrix_indexes.index
# (Make sure they are in the same order as the values in result_row['vicinity'])

# For demonstration, let's assume sample_ids are in a list:
sample_ids = list(binding_labels)  # adjust as needed if you have another identifier

# Convert the selected result_row into the desired long format
long_df = convert_results_to_long_format(result_row, binding_labels, sample_ids)

# If you want to adjust the threshold value (for instance, set it to 0),
# you can override the 'Threshold' column after conversion:
long_df['Threshold'] = 0

print(long_df.head())



