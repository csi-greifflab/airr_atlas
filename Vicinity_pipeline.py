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
sys.path.insert(0, '/doctorai/niccoloc/airr_atlas')

from Vicinity_analysis_class import Vicinity_analysis
from Vicinity_analysis_class import calculate_moran_index
from Vicinity_analysis_class import prepare_data_for_plotting
from Vicinity_analysis_class import run_ggplot_vicinity
from libpysal.weights import W

import argparse
from sklearn.preprocessing import scale

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--analysis_name', type=str, required=True, help='Name of the analysis')
    parser.add_argument('--df_junction_colname', type=str, default='junction_aa', help='Column name for junction aa')
    parser.add_argument('--df_affinity_colname', type=str, default='affinity', help='Column name for affinity')
    parser.add_argument('--input_metadata', type=str, required=True, help='Path to input metadata CSV file')
    parser.add_argument('--input_embeddings', type=str, required=True, help='Path to input embeddings file')
    parser.add_argument('--result_dir', type=str, default='./Vicinity_results', help='parent directory of results')
    parser.add_argument('--save_results', type=bool, default=True, help='Flag to save results (True/False)')
    parser.add_argument('--compute_LD', type=bool, default=False, help='Flag to compute LD (True/False)')
    parser.add_argument('--plot_results', type=bool, default=True, help='Flag to generate plots (True/False)')
    parser.add_argument('--chosen_metric', type=str, choices=['cosine', 'euclidean'], default='euclidean', help='Metric to use')
    parser.add_argument('--sample_size', type=int, default= 0 , help='Size of the max sample of each label')
    parser.add_argument('--LD_sample_size', type=int, default= 5000 , help='Number of seqs (X vs ALL) to check in the LD calculations')
    parser.add_argument('--precomputed_LD', type=str, required=False,help='path of the precomputed file.csv with LD results')
    return parser.parse_args()

def load_data(input_metadata, input_embeddings):
    if not os.path.exists(input_metadata):
        raise FileNotFoundError(f"Metadata file not found: {input_metadata}")
    if not os.path.exists(input_embeddings):
        raise FileNotFoundError(f"Embeddings file not found: {input_embeddings}")
    
    seqs = pd.read_csv(input_metadata, sep=None)
    seqs['id'] = np.arange(0, len(seqs))
    tensors = torch.load(input_embeddings).numpy()
    tensors_df = pd.DataFrame({
        'id': np.arange(0, len(tensors)),
        'embedding': list(tensors)
    })
    df = pd.merge(seqs, tensors_df, on='id')
    print("...Removing duplicated sequences ...")
    df = df[~df['junction_aa'].duplicated(keep=False)]
    df = df.reset_index(drop=True)
    df['id'] = np.arange(0, len(df))
    return df

def create_result_folder(res_folder):
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    else:
        print(f"Warning: Result folder {res_folder} already exists. Output files may be overwritten.")



def filter_data(df, max_junction_length=40, sample_size=10000, rand_seed=123, junction_aa_col='junction_aa', affinity_col='affinity'):
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



args = parse_arguments()
print("Starting analysis ...")

analysis_name = args.analysis_name
result_folder= os.path.join( args.result_dir, f"{analysis_name}/")
df_junction_colname= args.df_junction_colname
df_affinity_colname= args.df_affinity_colname
chosen_metric = args.chosen_metric

print(f" this is the result folder {result_folder}")

create_result_folder(result_folder)

df = load_data(args.input_metadata, args.input_embeddings)
id_index_sample = df['id']

if args.sample_size != 0 :
    df_sample_filt = filter_data(df,
                                 sample_size= args.sample_size,
                                 junction_aa_col=args.df_junction_colname,
                                 affinity_col=args.df_affinity_colname)
    id_index_sample=df_sample_filt['id']  # sampled_index



"""  ** Run Vicinity analysis ** """

max_neighbors = 1000 # This is the maximum number of neighbors you're interested in
part1 = np.arange(2, 304, 4)  # check in detail first 300 NN
part2 = np.arange(350, max_neighbors+1, 50)  # Second part: numbers from 300 to 1000 with steps of 50
neighbor_numbers = np.concatenate((part1, part2))

id_index_sample=df_sample_filt['id']  # indexes of 10k (depends on sample size)  each lb mb hb

#KNN vicinity
vicinity_analysis_instance = Vicinity_analysis(df,
                                                neighbor_numbers,
                                                id_index_sample,
                                                colname_affinity=df_affinity_colname,
                                                colname_junction=df_junction_colname,
                                                metric= chosen_metric)
vicinity_analysis_instance.run_analysis()  # This populates the necessary attributes
vicinity_analysis_instance.label_results

#ED radius vicinity --- # TODO : implement a better way of defining Custom distance radius thresholds
ED_radius = range(7, 25)  # Define your Euclidian distance radius to check -- should work for AB2
if chosen_metric == "cosine":
    ED_radius= np.arange(0,0.3,0.01)

print(ED_radius)
percentages_results, res_df, mean_num_points, LD1_res, LD2_res = vicinity_analysis_instance.perc_Euclidian_radius(ED_radius)
tmp_ed_sum=vicinity_analysis_instance.summary_results 


# ----------------- Save vicinity results ------
ED_filename= f"{result_folder}summary_results_ED_{analysis_name}.csv"
if args.save_results:
    vicinity_analysis_instance.save_to_pickle(f"{result_folder}Vicinity_{analysis_name}.pkl")
    vicinity_analysis_instance.summary_results.to_csv(ED_filename)

#------ compute the LD distance on a substet as comparator
np.random.seed(123)
#function to sample the max numb of sequences if they are below the chosen sample size
def sample_affinities(df, sample_size, df_affinity_colname ='affinity'):
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

chosen_sample_size=  args.LD_sample_size
LD_filename=f"{result_folder}d_mean1_summary_LD_{analysis_name}_{chosen_sample_size//1000}k.csv"
if args.compute_LD:
    rand_100k=sample_affinities(df, chosen_sample_size, df_affinity_colname).index
    max_LD=5
    d_res1,d_mean1 = prepare_data_for_plotting( df,max_LD, sampled_indices=rand_100k) # to get VICINITY percentages of LD dist 
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


