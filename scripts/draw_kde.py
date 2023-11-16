import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import torch
import seaborn as sns; sns.set()


parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("umap_path", type=str, help="UMAP path + /")
args = parser.parse_args()

#out_dir = "/doctorai/marinafr/2023/airr_atlas/analysis/output/"

#in_file_path = out_dir + args.embedding + '/'
in_file_path = args.umap_path

# List all files in the folder
file_list = os.listdir(in_file_path)

# Iterate through the files
for filename in file_list:
    if 'UMAP' in filename and ('pt' in filename or 'pkl' in filename or 'npy' in filename):
        output_file = in_file_path + 'KDE_' + filename.strip('UMAP_').split('.')[0] + '.png'

        # Load embedding file
        if "pt" in filename:
            data = torch.load(in_file_path+filename).numpy()
        elif "pkl" in filename:
            data = pd.read_pickle(in_file_path+filename)#.values
        elif "npy" in filename:
            data = np.load(in_file_path+filename)
        print(f"Read {in_file_path+filename} with {data.shape[0]} sequences")

        plt.figure(figsize=(12, 12))
        sns.kdeplot(data=data, x=data.iloc[:,0].values, y=data.iloc[:,1].values, fill=True)
        plt.xlabel('UMAP_1', fontsize=20)
        plt.ylabel('UMAP_2', fontsize=20)
        plt.savefig(f'{output_file}')
