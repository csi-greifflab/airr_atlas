import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openTSNE
#open tsne version print
print(f"openTSNE version: {openTSNE.__version__}")

#set working directory as /doctorai/niccoloc
from rapidfuzz.distance import Levenshtein as RapidfuzzLevenshtein
import os

from tqdm import tqdm
os.chdir('/doctorai/niccoloc')
from my_functions.get_W2 import get_w2
from collections import OrderedDict

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
post_df = pd.read_csv('/doctorai/niccoloc/post_selection_1M_pgen_NEW.csv')
# Load ireceptor metadata and embeddings
ireceptor_metadata_df = pd.read_csv(ireceptor_metadata)

#sequence aa column name in postselection df is 'cdr3_aa'
post_df['specificity'] = 'Post-selection'
ireceptor_metadata_df['specificity'] = 'iReceptor'
post_df.rename(columns={'amino_acid': 'sequence_aa'}, inplace=True)
ireceptor_metadata_df.rename(columns={'junction_aa': 'sequence_aa'}, inplace=True)
#sample 20k sequences from ireceptor metadata and postselection metadata for faster computation
ireceptor_metadata_df = ireceptor_metadata_df.sample(n=20000, random_state=42).reset_index(drop=True)
post_df = post_df.sample(n=20000, random_state=42).reset_index(drop=True)

#concatenate the ireceptor metadata with postselection metadata and ag_df 
combined_df = pd.concat([ag_df, post_df, ireceptor_metadata_df], ignore_index=True)



# compute pairwise Levenshtein distances between every pair of specificities in ag_df
import numpy as np
import pandas as pd
from collections import OrderedDict
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

 


# --- Configuration ---
seq_col = "sequence_aa"
spec_col = "specificity"

# --- Group sequences by specificity ---
spec_groups = combined_df.groupby(spec_col)[seq_col].apply(list).to_dict()
spec_names = list(spec_groups.keys())
n_specs = len(spec_names)



print(f"Computing pairwise Levenshtein distances for {n_specs} specificities...")

# --- Initialize results containers ---
pairwise_results = OrderedDict()

# --- Compute pairwise distances ---
for i, spec1 in enumerate(tqdm(spec_names, desc="Specificities", unit="spec")):
    seqs1 = spec_groups[spec1]
    for j, spec2 in enumerate(spec_names):
        if j < i:
            continue  # avoid recomputation, upper triangle only

        seqs2 = spec_groups[spec2]
        n1, n2 = len(seqs1), len(seqs2)

        print(f"[{i+1}/{n_specs}] '{spec1}' ({n1}) x '{spec2}' ({n2})")

        # Compute the distance matrix
        mat = np.empty((n1, n2), dtype=float)
        for ii, s1 in enumerate(seqs1):
            for jj, s2 in enumerate(seqs2):
                mat[ii, jj] = Levenshtein.distance(s1, s2)

        # Compute basic statistics
        mean_dist = float(mat.mean()) if mat.size else None
        std_dist = float(mat.std()) if mat.size else None

        result = {
            "matrix": mat,
            "mean": mean_dist,
            "std": std_dist,
            "shape": mat.shape
        }
        pairwise_results[(spec1, spec2)] = result

        # Add the symmetric entry for convenience
        if spec1 != spec2:
            pairwise_results[(spec2, spec1)] = {
                "matrix": mat.T,
                "mean": mean_dist,
                "std": std_dist,
                "shape": mat.T.shape
            }

print("✅ Pairwise Levenshtein distance computation completed.")

intra_results = {k: v for k, v in pairwise_results.items() if k[0] == k[1]}
inter_results = {k: v for k, v in pairwise_results.items() if k[0] != k[1]}


intra_mean = np.mean([v['mean'] for v in intra_results.values() if v['mean'] is not None])
inter_mean = np.mean([v['mean'] for v in inter_results.values() if v['mean'] is not None])
print(f"Mean intra-specificity distance: {intra_mean:.2f}")
print(f"Mean inter-specificity distance: {inter_mean:.2f}")

#check percentage of intra distances that are not zero
intra_nonzero_counts = []
for k, v in intra_results.items():
    mat = v['matrix']
    nonzero_count = np.sum(mat != 0)
    total_count = mat.size
    intra_nonzero_counts.append((nonzero_count, total_count))

total_nonzero = sum([count[0] for count in intra_nonzero_counts])
total_counts = sum([count[1] for count in intra_nonzero_counts])
percentage_nonzero = (total_nonzero / total_counts) * 100 if total_counts > 0 else 0
print(f"Percentage of non-zero intra-specificity distances: {percentage_nonzero:.2f}%")

#plot heatmap of of both intra and inter specificity distances
import matplotlib.pyplot as plt
import seaborn as sns   
# Create a matrix for heatmap
heatmap_matrix = np.zeros((n_specs, n_specs))
for i, spec1 in enumerate(spec_names):
    for j, spec2 in enumerate(spec_names):
        heatmap_matrix[i, j] = pairwise_results[(spec1, spec2)]['mean']
# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_matrix, xticklabels=spec_names, yticklabels=spec_names, cmap="viridis", annot=True, fmt=".5f")
plt.title("Mean Levenshtein Distances Between Specificities")
plt.xlabel("Specificity")
plt.ylabel("Specificity")
plt.tight_layout()
plt.show()
plt.savefig('/doctorai/niccoloc/mean_levenshtein_distance_heatmap_LEV_NORM.png')






import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Extract unique specificities, sorted
specs = sorted(set([k[0] for k in pairwise_results.keys()]))
specs= ['HIV',   'TG2+', 'covid', 'ebola', 'influenza', 'malaria','iReceptor','Post-selection']
n = len(specs)

# Figure setup
fig = plt.figure(figsize=(1.8*n, 1.8*n))
gs = gridspec.GridSpec(n, n, wspace=0.5, hspace=0.5)

#match the specs original names to shorter names for better visualization in the plot
# specs = ['HIV', 'TG2+', 'covid', 'ebola', 'influenza', 'malaria']
new_names = ["HIV","TG2+","SARS-CoV2 RBD","Ebola", "Influenza","Malaria","iReceptor","Post-selection"]
short_names = { 
    'HIV': 'HIV',
    'TG2+': 'TG2',
    'covid': 'SARS-CoV-2\nRBD',
    'ebola': 'Ebola',
    'influenza': 'Influenza',
    'malaria': 'Malaria',
    'iReceptor': 'iReceptor',
    'Post-selection': 'Post-selection\nsimulated'
}

#replace specs with short names
specs1 = [short_names.get(s, s) for s in specs]

for i, s1 in enumerate(specs):
    for j, s2 in enumerate(specs):
        ax = fig.add_subplot(gs[i, j])

        # Flatten LD values for the pair
        data = pairwise_results[(s1, s2)]["matrix"].flatten()
        mean = pairwise_results[(s1, s2)]["mean"]
        std= pairwise_results[(s1, s2)]["std"]

        # Plot a tiny histogram (1D distribution)
        ax.hist(data, bins=25, density=True)

        # Mean annotation - left side
        ax.text(0.05, 0.85, f"mean:\n{mean:.2f}",
                ha="left", va="center",
                transform=ax.transAxes, 
                fontsize=8,
                color='blue')
        # Std annotation - right side 
        ax.text(0.95, 0.85, f"std:\n{std:.2f}",
                ha="right", va="center", 
                transform=ax.transAxes, fontsize=8,
                color='red')

        # Remove ticks everywhere
        # ax.set_xticks([])
        # ax.set_yticks([])
        #x limits from 0 to 30
        ax.set_xlim(0, 30)
        # ax.set_ylim(0, 0.20)
        #add y labels to only the first column as density
        # if j == 0:
        #     ax.set_ylabel("Density", rotation=0, labelpad=30, va='center')

        # Only label left column and top row
        ag_names1 = short_names.get(s1, s1)
        ag_names2 = short_names.get(s2, s2)
        # Only label the first column with Density
        if j == 0:
            ax.set_ylabel("Density", rotation=90, labelpad=5, va='center')

        # Move antigen names to the left spine instead and pad it a little
        if j == 0:
            ax.text(-1.2, 0.5, ag_names1, rotation=0, va='center', 
                ha='center', transform=ax.transAxes)
        if i == 0:
            ax.set_title(ag_names2, rotation=0, pad=20)
        #if its a diagonal combination add a purple border
        if i == j:
            for spine in ax.spines.values():
                spine.set_edgecolor('purple')
                spine.set_linewidth(2)  

#add shared x label as "Levenshtein Distance"
fig.text(0.5, 0.04, 'Levenshtein Distance (aa)', ha='center', va='center', fontsize=14)
#add shared y label as "Density
plt.tight_layout()
plt.savefig('/doctorai/niccoloc/pairwise_levenshtein_distance_distributions_LEV_NORM2.png', bbox_inches='tight')
plt.clf()


#save the pairwise_results in the best format
import pickle
with open('/doctorai/niccoloc/pairwise_levenshtein_distance_results_LEV_NORM.pkl', 'wb') as f:
    pickle.dump(pairwise_results, f)

#load the pairwise_results from the pickle file
import pickle
with open('/doctorai/niccoloc/pairwise_levenshtein_distance_results_LEV_NORM.pkl', 'rb') as f:
    pairwise_results = pickle.load(f)
