import torch
import sys
#sys.path.append("/doctorai/marinafr/2023/airr_atlas/wang_paper/libraries/")
import argparse
import numpy as np
from sklearn.preprocessing  import MinMaxScaler
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("input_path", type=str, help="Input path + filename.pt")
parser.add_argument("output_path", type=str, help="Output path + filename.pt")
args = parser.parse_args()

input_file = args.input_path
output_file = args.output_path

# Load the embeddings
X = torch.load(input_file).numpy()

# Scaler
scaler = MinMaxScaler()
scaler.fit(X)
X_norm = scaler.transform(X)

# PCA to reduce dimensionality
pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X_norm)

# Save the output
torch.save(X_reduced, output_file)
