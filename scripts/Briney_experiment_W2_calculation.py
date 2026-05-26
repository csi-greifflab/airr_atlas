import numpy as np
import argparse
import pandas as pd
from scipy import linalg
import torch
import os
from tqdm import tqdm
import cupy as cp
import itertools
import torch

def cupy_sqrtm(A, disp=True):

    # Ensure array is on the GPU
    A = cp.asarray(A)
    # Compute the eigenvalue decomposition
    eigenvalues, eigenvectors = cp.linalg.eigh(A)
    # Compute the square root of the eigenvalues
    sqrt_eigenvalues = cp.sqrt(eigenvalues)
  #  print(cp.diag(sqrt_eigenvalues))
    # Reconstruct the matrix square root
    sqrtm_A = eigenvectors @ cp.diag(sqrt_eigenvalues) @ cp.linalg.inv(eigenvectors)
    print(sqrtm_A)
    if disp == False:
        # Calculate the estimated error
        A_estimated = sqrtm_A @ sqrtm_A
        error = A - A_estimated
        frobenius_norm = cp.linalg.norm(error, ord='fro')
        
        return sqrtm_A, frobenius_norm
    else:
        return sqrtm_A

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1:    The mean of the activations of preultimate layer of the
               CHEMNET (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2:    The mean of the activations of preultimate layer of the
               CHEMNET (like returned by the function 'get_predictions')
               for real samples.
    -- sigma1: The covariance matrix of the activations of preultimate layer
               of the CHEMNET (like returned by the function 'get_predictions')
               for generated samples.
    -- sigma2: The covariance matrix of the activations of preultimate layer
               of the CHEMNET (like returned by the function 'get_predictions')
               for real samples.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = cp.atleast_1d(mu1)
    mu2 = cp.atleast_1d(mu2)

    sigma1 = cp.atleast_2d(sigma1)
    sigma2 = cp.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(cp.asnumpy(sigma1.dot(sigma2)),disp=False)
    covmean = cp.asarray(covmean)
    if not cp.isfinite(covmean).all():
        offset = cp.eye(sigma1.shape[0]) * eps
        covmean = cp.asarray(linalg.sqrtm(cp.asnumpy((sigma1 + offset).dot(sigma2 + offset))))
    # numerical error might give slight imaginary component
    if cp.iscomplexobj(covmean):
        if not cp.allclose(cp.diagonal(covmean).imag, 0, atol=1e-3):
            m = cp.max(cp.abs(covmean.imag))
            #raise ValueError("Imaginary component {}".format(m))
            print("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = cp.trace(covmean)
  #  print(diff,tr_covmean)

    return diff.dot(diff) + cp.trace(sigma1) + cp.trace(sigma2) - 2 * tr_covmean


def get_w2(combination):
    act1 = batch_dict[combination[0]]
    act2 = batch_dict[combination[1]]
    """Calculate w2 between two sets

    Args:
        act1: First set
        act2: Second set

    Returns:
        float: The FCD score
    """

    mu1 = cp.mean(act1, axis=0)
    sigma1 = cp.cov(act1.T)

    mu2 = cp.mean(act2, axis=0)
    sigma2 = cp.cov(act2.T)
    fcd_score = calculate_frechet_distance(
        mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2
    )
    return (combination[0],combination[1],fcd_score)


parser = argparse.ArgumentParser()
parser.add_argument("--metadata", help="path to metadata")
parser.add_argument("--embed", help="path to embedding")
args = parser.parse_args()

out_file_name = '/cluster/work/projects/ec195/evgeniie/'+'w2_AB2'+args.metadata.split('/')[-1]+'.csv'
print('File will be saved as ',out_file_name)
metadata = pd.read_csv(args.metadata)
briney_data = torch.load(args.embed).numpy()
print('DATA LOADED')

#Getting all possible combinations for batches
all_patients = list(set(metadata['batch'].tolist()))
print(len(all_patients))
combos = list(itertools.combinations(all_patients, 2))
print(len(combos))
#Storing all batches in special dictionary
batch_dict = dict()
for patient in all_patients:
        first_index = metadata.index[metadata['batch'] == patient].min()
        last_index = metadata.index[metadata['batch'] == patient].max()
#       print(first_index,last_index)
        batch_dict[patient] = briney_data[int(first_index):int(last_index+1)]
print('DATA STORED')
#Getting all possible combinations for batches

correlations = []
#Calculating w2 pairwise distances for all batches
for combo in tqdm(combos):
    a = get_w2(combo)
    correlations.append(a)
print('ENDED')
correlations_df = pd.DataFrame(correlations, columns=['Sample1', 'Sample2', 'w2_distance'])

correlations_df.to_csv(out_file_name,index=True)
