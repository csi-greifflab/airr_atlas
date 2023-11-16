import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import torch
import seaborn as sns; sns.set()


parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("embedding", type=str, help="baseline, prott5, etc")
args = parser.parse_args()

out_dir = "/doctorai/marinafr/2023/airr_atlas/analysis/output/"
#out_dir = '/home/marina/Documents/oslo/airr_atlas/output/'

in_file_path = out_dir + args.embedding + '/'

# List all files in the folder
file_list = os.listdir(in_file_path)

# Iterate through the files
for filename in file_list:
    if 'UMAP' in filename and ('pt' in filename or 'pkl' in filename or 'npy' in filename):
        output_file = in_file_path + filename.split('.')[0] + '.png'

        # Load embedding file
        if "pt" in filename:
            data = torch.load(in_file_path+filename).numpy()
        elif "pkl" in filename:
            data = pd.read_pickle(in_file_path+filename)#.values
        elif "npy" in filename:
            data = np.load(in_file_path+filename)
        print(f"Read {in_file_path+filename} with {data.shape[0]} sequences")

        plt.figure(figsize=(12, 12))
        scatter = sns.scatterplot(data=data, x=data.iloc[:,0].values, y=data.iloc[:,1].values, s=3, linewidth=0)
        plt.xlabel('UMAP_1', fontsize=20)
        plt.ylabel('UMAP_2', fontsize=20)
        plt.savefig(f'{output_file}')
