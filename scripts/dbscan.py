import sys
from sklearn.cluster import DBSCAN
from sklearn.preprocessing  import MinMaxScaler
import numpy as np
import argparse
import torch
import pandas as pd


np.random.seed(42)
num_samples = 100000

parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("input_path", type=str, help="Fasta path + filename.pt")
parser.add_argument("output_path", type=str, help="Output path + filename.pkl")
args = parser.parse_args()

# Load the embeddings
df_pt = torch.load(args.input_path).numpy()

# Get a subset
df_sub = df_pt[np.random.choice(df_pt.shape[0], num_samples, replace=False)]

# Normalize the embeddings
scaler = MinMaxScaler()
scaler.fit(df_sub)
X_norm = scaler.transform(df_sub)

# Run DBSCan and save the labels
dbscan = DBSCAN(eps=4, min_samples=4, metric='euclidean')
labels = dbscan.fit_predict(X_norm)
pd.DataFrame(labels).to_pickle(args.output_path)
print(f"Saved {args.output_path} with {len(labels)} labels")

