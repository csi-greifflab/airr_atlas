import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#set working directory as /doctorai/niccoloc
import os
os.chdir('/doctorai/niccoloc')
from my_functions.get_W2 import get_w2

#NEW PATHS
mat_dict = {}
diag_dict = {}
std_diagonal_dict = {}

ireceptor_metadata = '/doctorai/niccoloc/ireceptor_NEW_final_onlyseqid.csv'
irecep_esm2 = '/doctorai/niccoloc/ireceptor/esm2_t33_650M_UR50D/mean_pooled/ireceptor_2M_esm2_t33_650M_UR50D_mean_pooled_layer_33.npy'
irecep_ab2 = '/doctorai/niccoloc/ireceptor/antiberta2-cssp/mean_pooled/ireceptor_2M_antiberta2-cssp_mean_pooled_layer_16.npy'
ag_metadata = '/doctorai/userdata/airr_atlas/data/sequences/bcr/ALL_ANTIGENS/antigen_specific_df_2025.csv'
ag_esm2 = '/scratch/niccoloc/ag_dataset/esm2_t33_650M_UR50D/embeddings/ag_dataset_esm2_t33_650M_UR50D_embeddings_layer_33.npy'
ag_ab2 = '/scratch/niccoloc/ag_dataset/antiberta2-cssp/embeddings/ag_dataset_antiberta2-cssp_embeddings_layer_16.npy'
postselection= '/doctorai/niccoloc/postselection_1M/esm2_t33_650M_UR50D/mean_pooled/postselection_1M_esm2_t33_650M_UR50D_mean_pooled_layer_33.npy'

ag_OHE='/doctorai/niccoloc/w2_AG_DATA_OHE/w2_AG_DATA_OHE_OHE.pt' 
irecep_OHE='/doctorai/niccoloc/w2_irecep_OHE/w2_irecep_OHE_OHE.pt' 

def load_data(input_metadata, input_embeddings,idx_reference ,df_junction_colname='cdr3_aa', df_affinity_colname='binding_label', filter_out='lb'):
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
    df = df[~df[df_junction_colname].duplicated(keep=False)]
    #filter out lb
    df = df[~df[df_affinity_colname].isin([filter_out])]
    print(f"Number of sequences in the dataset after de-deuplication: {len(df)}, per class: {df[df_affinity_colname].value_counts()}")
    embeddings = tensors[df['tensor_id'].values]
    df = df.reset_index(drop=True)
    df['id'] = np.arange(0, len(df))

    return df , embeddings

# Load antigen metadata and index first
ag_df = pd.read_csv(ag_metadata)
ag_idx = pd.read_csv('/scratch/niccoloc/ag_dataset/antiberta2-cssp/antigen_specific_2025_idx.csv')
ag_df = ag_df.merge(ag_idx, on='sequence_id', how='left')
# Load postselection metadata and embeddings
post_df = pd.read_csv('/doctorai/niccoloc/post_selection_1M_pgen_NEW.csv')
# Load ireceptor metadata and embeddings
ireceptor_metadata_df = pd.read_csv(ireceptor_metadata)




# -----------------------------
# Base metadata (unchanged)
# -----------------------------
ireceptor_metadata = '/doctorai/niccoloc/ireceptor_NEW_final_onlyseqid.csv'
ag_metadata = '/doctorai/userdata/airr_atlas/data/sequences/bcr/ALL_ANTIGENS/antigen_specific_df_2025.csv'
ag_idx_path   = '/scratch/niccoloc/ag_dataset/antiberta2-cssp/antigen_specific_2025_idx.csv'
post_meta     = '/doctorai/niccoloc/post_selection_1M_pgen_NEW.csv'  #

# -----------------------------
# Model-specific embedding paths
# -----------------------------
MODEL_PATHS = {
    'esm2': {
        'ag':   '/scratch/niccoloc/ag_dataset/esm2_t33_650M_UR50D/embeddings/ag_dataset_esm2_t33_650M_UR50D_embeddings_layer_33.npy',
        'irec': '/doctorai/niccoloc/ireceptor/esm2_t33_650M_UR50D/mean_pooled/ireceptor_2M_esm2_t33_650M_UR50D_mean_pooled_layer_33.npy',
        'post': '/doctorai/niccoloc/postselection_1M/esm2_t33_650M_UR50D/mean_pooled/postselection_1M_esm2_t33_650M_UR50D_mean_pooled_layer_33.npy',
    },
    'ab2': {
        'ag':   '/scratch/niccoloc/ag_dataset/antiberta2-cssp/embeddings/ag_dataset_antiberta2-cssp_embeddings_layer_16.npy',
        'irec': '/doctorai/niccoloc/ireceptor/antiberta2-cssp/mean_pooled/ireceptor_2M_antiberta2-cssp_mean_pooled_layer_16.npy',
        'post': '/doctorai/niccoloc/postselection_1M/antiberta2-cssp/mean_pooled/postselection_1M_antiberta2-cssp_mean_pooled_layer_16.npy',  # add a path here if you have postselection for ab2
    },
    'ohe': {
        'ag':   '/doctorai/niccoloc/w2_AG_DATA_OHE/w2_AG_DATA_OHE_OHE.pt',
        'irec': '/doctorai/niccoloc/w2_irecep_OHE/w2_irecep_OHE_OHE.pt',
        'post': '/doctorai/niccoloc/w2_POSTSEL_OHE/w2_POSTSEL_OHE_OHE.pt',  # add if you have OHE postselection
    }
}


# -----------------------------
def load_embeddings(path):
    if path is None:
        return None
    if path.endswith('.npy'):
        return np.load(path, mmap_mode='r')
    if path.endswith('.pt'):
        import torch
        t = torch.load(path, map_location='cpu')
        return t.numpy() if hasattr(t, 'numpy') else t.detach().cpu().numpy()
    raise ValueError(f"Unsupported embedding file type: {path}")



EMB_TYPE = 'ab2'   # <-- change to 'ab2' or 'ohe'
EMB_TYPE = 'esm2'   # <-- change to 'ab2' or 'ohe'

ag_embeddings_path    = MODEL_PATHS[EMB_TYPE]['ag']
irecep_embeddings_path = MODEL_PATHS[EMB_TYPE]['irec']
postselection_path    = MODEL_PATHS[EMB_TYPE]['post']


ag_embeddings  = load_embeddings(ag_embeddings_path)
irecp_embeddings =  load_embeddings(irecep_embeddings_path)
postselection_embeddings = load_embeddings(postselection_path)

# Prepare antigen embeddings dictionary
ag_embeddings_dict_X = {}
for antigen, group in ag_df.groupby('specificity'):
    print(f'Processing antigen: {antigen}')
    print(f'Number of sequences: {(group)}')
    ag_embeddings_dict_X[antigen] = ag_embeddings[group['index'].values]

# Remove HIV from the dictionary
ag_embeddings_dict_X.pop('HIV', None)

# Deduplicate ireceptor metadata and add to dictionary
ireceptor_metadata_df_dedup = ireceptor_metadata_df.drop_duplicates(subset=['junction_aa'])
batch_df = ireceptor_metadata_df_dedup.sample(n=100000, random_state=43)
ag_embeddings_dict_X['irecpt'] = irecp_embeddings
# ag_embeddings_esm2_dict['irecpt'] = irecp_embeddings[batch_df.index.values]  # If you want to sample

# Clean postselection data and add to dictionary
post_df = post_df[post_df['amino_acid'].str.len() >= 5]
post_df = post_df.drop_duplicates(subset=['amino_acid'])
batch_df = post_df.sample(n=100000, random_state=43).reset_index()
ag_embeddings_dict_X['postselection_1M'] = postselection_embeddings

# # Add tz embeddings to dictionary
# ag_embeddings_dict_X['tz'] = embeddings_hb



print(ag_embeddings_dict_X.keys())
ag_embeddings_dict_X['covid'].shape 
ag_embeddings_dict_X['irecpt'].shape 



def compute_W2_pairwise(x1, x2, x1_name: str, x2_name: str, device: str = 'cpu'):
    
    # w2_distance = get_w2(x1, x2, device=device)
    w2_distance = get_w2(x1, x2 )
    results = {
        "label_x1": x1_name,
        "label_x2": x2_name,
        "w2_distance": float(w2_distance)
    }
    return results


import itertools
# Loop through unique pairs
names = list(ag_embeddings_dict_X.keys())
n = len(names)
dist_matrix = np.zeros((n, n))


for name1, name2 in itertools.combinations(names, 2):
    # #skip if name1 or name2 is HIV
    # if name1 == 'HIV' or name2 == 'HIV':
    #     continue
    print(f'Computing W2 distance between {name1} and {name2}, shapes: {ag_embeddings_dict_X[name1].shape} vs {ag_embeddings_dict_X[name2].shape}')
    results = compute_W2_pairwise(
        ag_embeddings_dict_X[name1],
        ag_embeddings_dict_X[name2],
        x1_name=name1,
        x2_name=name2,
        device='cpu'
    )
    print(f'W2 distance between {name1} and {name2}: {results["w2_distance"]}')
    dist = results["w2_distance"]
    i, j = names.index(name1), names.index(name2)
    dist_matrix[i, j] = dist
    dist_matrix[j, i] = dist  # symmetry

#round the values to 3 decimal places
# dist_matrix = dist_matrix.round(3)

#get the diagonal

import itertools
from tqdm import trange
from tqdm import tqdm

from joblib import Parallel, delayed
# Loop through unique pairs
names = list(ag_embeddings_dict_X.keys())
n = len(names)


ag_repeat_tests = {}
def split_and_compute_w2(ag, ag_embeddings, repeats=10, seed=42, batch_size=None):
    rng = np.random.default_rng(None)
    n_ag = ag_embeddings.shape[0]
    w2_results = []
    for repeat in range(repeats):
        if ag in [  'irecpt', 'postselection_1M']:
        #sample 20k
            first_sample = rng.choice(n_ag , size=20000, replace=False)
            ag_embeddings = ag_embeddings[first_sample]  
            n_ag = ag_embeddings.shape[0]               
        idx = rng.permutation(n_ag)
        mid = n_ag // 2
        idx_mid1 = idx[:mid]
        idx_mid2 = idx[mid:]
        if batch_size == None:
            idx_batch1 = idx_mid1
        else:
            idx_batch1 = idx_mid1[:min(batch_size, len(idx_mid1))]

        ag1 = ag_embeddings[idx_batch1]
        ag2 = ag_embeddings[idx_mid2]
        #print ag size 
        print(f"Repeat {repeat+1}: ag1 size = {ag1.shape[0]}, ag2 size = {ag2.shape[0]}")
        name1_ag = f"{ag}_1"
        name2_ag = f"{ag}_2"
        res = compute_W2_pairwise(
            ag1,
            ag2,
            x1_name=name1_ag,
            x2_name=name2_ag,
            device='cpu'
        )
        w2_results.append(res["w2_distance"])
    mean_w2 = np.mean(w2_results)
    std_w2 = np.std(w2_results, ddof=1)
    return ag, (mean_w2, std_w2, w2_results)
ag_repeat_tests = {}

#

# global_rng = np.random.default_rng(42)

names = list(ag_embeddings_dict_X.keys())
# for batch_size in [800, 1000, 1200, None]:
batch_sizes= [ 1200 , 10000, 100000, None]
batch_sizes= [  10000]
batch_sizes= [  None]


for batch_size in batch_sizes:
    print(f"\n--- Batch size: {batch_size} ---")
    results = Parallel(n_jobs=80)(
        delayed(split_and_compute_w2)(ag, ag_embeddings_dict_X[ag], 10, 42, batch_size=batch_size) for ag in names
    )
    for ag, result in results:
        ag_repeat_tests[(ag, batch_size)] = result
        print(f"{ag} (batch {batch_size}): mean W2={result[0]:.4f}, std={result[1]:.4f} over 10 splits")

#round the values to 3 decimal places
# dist_matrix = dist_matrix.round(3)

# Extract only the mean and std values for each batch from ag_repeat_tests
mean_std_per_batch = {
    batch: {ag: (vals[0], vals[1]) for (ag, b), vals in ag_repeat_tests.items() if b == batch}
    for batch in set(b for (_, b) in ag_repeat_tests.keys())
}
# Example: mean_std_per_batch[1200]['covid'] -> (mean_w2, std_w2) for batch size 1200 and antigen 'covid'
print(mean_std_per_batch)

# diagonal_1200 = np.array([mean_std_per_batch[1200][ag][0] for ag in names])
diagonal_10k = np.array([mean_std_per_batch[None][ag][0] for ag in names])
std_diagonal_10k = np.array([mean_std_per_batch[None][ag][1] for ag in names])



#ag_repeat_tests Wrap into a DataFrame
df_ag_repeat = pd.DataFrame.from_dict(ag_repeat_tests, orient='index', columns=['mean_w2', 'std_w2','all_w2s'])

#sort by name index alphabetically
df_ag_repeat = df_ag_repeat.sort_index()

# all_w2s as dataframe with coluns as the antigen and rows as the repeats 
df_ag_repeat_all = pd.DataFrame(df_ag_repeat['all_w2s'].tolist(), columns=[f"repeat_{i+1}" for i in range(len(df_ag_repeat['all_w2s'].iloc[0]))], index=df_ag_repeat.index).T


mat_dict[EMB_TYPE] = dist_matrix
diag_dict[EMB_TYPE] = diagonal_10k
std_diagonal_dict[EMB_TYPE] = std_diagonal_10k






#choose again the emb type
EMB_TYPE = 'ab2'   # <-- change to 'ab2' or 'ohe'
EMB_TYPE = 'esm2'   # <-- change to 'ab2' or '
dist_matrix = mat_dict[EMB_TYPE]
diagonal_10k = diag_dict[EMB_TYPE]
std_diagonal_10k = std_diagonal_dict[EMB_TYPE]

#substitute the diagonal with the diagonal_1200 object

dist_matrix[np.diag_indices(n)] = diagonal_10k
std_diagonal_10k

# Wrap into a DataFrame
w2_df = pd.DataFrame(dist_matrix, index=names, columns=names)
#filter out tz
print(w2_df)

#get number of items for each antigen
antigen_counts = {name: ag_embeddings_dict_X[name].shape[0] for name in names}
print(antigen_counts)

# Update names to include counts
names_with_counts = [f"{name} (n={antigen_counts[name]})" for name in names]



#drop tz 
# w2_df = w2_df.drop(index='tz (n=49721)', columns='tz (n=49721)')




#plot and save the distance matrix

# plt.figure(figsize=(10, 8))
# sns.heatmap(w2_df, annot=True, fmt=".3f", cmap='viridis', square=True, cbar_kws={"shrink": .8}, annot_kws={"fontsize": 9})
# plt.title('Wasserstein Distance Matrix', fontsize=16)
# plt.xticks(rotation=45, ha='right', fontsize=10)
# plt.yticks(rotation=0, fontsize=10)
# plt.tight_layout()

# plt.savefig(f"/doctorai/niccoloc/w2_distance_matrix_{MODEL}.png", dpi=300)
# plt.savefig('/doctorai/niccoloc/w2_distance_matrix_TEST.png', dpi=300)
# plt.savefig('/doctorai/niccoloc/w2_distance_matrix_TEST_100k.png', dpi=300)
# plt.savefig('/doctorai/niccoloc/w2_distance_matrix_b1.png', dpi=300)
# plt.savefig('/doctorai/niccoloc/w2_distance_matrix_OHE.png', dpi=300)
# plt.clf()
# Save the DataFrame to a CSV file

# If you have your embeddings dict with counts:
antigen_counts = {name: ag_embeddings_dict_X[name].shape[0] for name in names}

# Update names to include counts
names_with_counts = [f"{name} (n={antigen_counts[name]})" for name in names]

# Build a mapping from original names -> names_with_counts
name_map = dict(zip(names, names_with_counts))

# Apply the mapping to rows/cols
w2_df.index = [name_map[i] for i in w2_df.index]
w2_df.columns = [name_map[j] for j in w2_df.columns]
new_names_id = ["TG2","SARS-CoV-2 RBD","Ebola", "Influenza","Malaria","iReceptor","Post-selection\nsimulated"]
# new_names_id = ["TG2+","CovAb-Dab (RBD)","Ebola", "Influenza","Malaria","iReceptor","Post-selection"]

#concatenate the new names with the counts
new_names = [f"{new_names_id[i]}\n(n={antigen_counts[names[i]]})" for i in range(len(new_names_id))]

w2_df.index =  new_names
w2_df.columns = new_names

# Start from the original order (`names`) where std_diagonal_10k is aligned:
std_series = pd.Series(std_diagonal_10k, index=names).rename(index=name_map)
# Align to w2_df (after dropping tz / renaming)
std_series = std_diagonal_10k

# -----------------------
# Build string annotations: value on off-diagonal, value\n(std) on diagonal
# -----------------------

from joblib import Parallel, delayed
import os
from tqdm import tqdm

vals = w2_df.values  # numeric for colormap
r, c = vals.shape
annot = np.empty((r, c), dtype=object)

for i in range(r):
    for j in range(c):
        if i == j:
            annot[i, j] = f"{vals[i, j]:.3f}\nstd:{std_series[i]:.3f}"
        else:
            annot[i, j] = f"{vals[i, j]:.3f}"

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    w2_df,
    annot=annot,      # string matrix
    fmt='',           # important: use strings as-is
    cmap='viridis',
    #add black border to each cell
    linecolor='black',
    linewidth=0.35,
    square=True,
    cbar_kws={"shrink": .8},
    annot_kws={"fontsize": 10}  # slightly smaller to fit two lines
)
plt.title(f'Ag-specific W2 distance matrix: ({EMB_TYPE})', fontsize=16)
plt.title(f'{EMB_TYPE.upper()}', fontsize=16)
#slightly adjust tick to the left
plt.xticks(rotation=45, ha='right', fontsize=11 , rotation_mode='anchor', position=(0,0))
plt.yticks(rotation=0, fontsize=11 )
plt.tight_layout()

# (Nice touch) Bold the diagonal annotation texts
texts = ax.texts
k = 0
for i in range(r):
    for j in range(c):
        if i == j:
            texts[k].set_fontweight('bold')
        k += 1

out_path = f"/doctorai/niccoloc/w2_distance_matrix_{EMB_TYPE}.png"
plt.savefig(out_path, dpi=300)
plt.clf()
print(f"Saved figure to {out_path}")


#============================variance estimation with repeats ==========================

from sklearn.preprocessing import MinMaxScaler

# 2. Flatten and compute statistics
# -------------------------
#esm2 distance matrix
esm2_mat= mat_dict['esm2']
esm2_diag= diag_dict['esm2']
esm2_mat[np.diag_indices(n)] = esm2_diag
esm2_flat = esm2_mat.flatten()
#ab2 distance matrix
ab2_mat = mat_dict['ab2']
ab2_diag = diag_dict['ab2']
ab2_mat[np.diag_indices(n)] = ab2_diag
ab2_flat = ab2_mat.flatten()


def describe(values):
    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "var": np.var(values),
        "min": np.min(values),
        "max": np.max(values),
        "range": np.max(values) - np.min(values),
        "cv": np.std(values) / np.mean(values)
    }

esm2_stats = describe(esm2_flat)
ab2_stats = describe(ab2_flat)

print("=== Raw descriptive statistics ===")
print("ESM2:", esm2_stats)
print("AB2 :", ab2_stats)

# -------------------------
# 3. Min–Max normalization
# -------------------------
scaler = MinMaxScaler()
esm2_mm = scaler.fit_transform(esm2_flat.reshape(-1, 1)).flatten()
ab2_mm = scaler.fit_transform(ab2_flat.reshape(-1, 1)).flatten()

esm2_mm_var = np.var(esm2_mm)
ab2_mm_var = np.var(ab2_mm)

print("\n=== Min–Max Normalized Variances ===")
print(f"ESM2 variance: {esm2_mm_var:.6f}")
print(f"AB2 variance:  {ab2_mm_var:.6f}")

# -------------------------
# 4. Quantitative comparison
# -------------------------
ratio_raw = ab2_stats["var"] / esm2_stats["var"]
ratio_mm = ab2_mm_var / esm2_mm_var

print("\n=== Variance comparison summary ===")
print(f"AB2 / ESM2 variance ratio (raw): {ratio_raw:.1f}x")
print(f"AB2 / ESM2 variance ratio (normalized): {ratio_mm:.2f}x")

