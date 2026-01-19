import numpy as np
import pandas as pd
from scipy import linalg
from multiprocessing import Pool
import os
try:
  import cupy as cp
except ImportError:
  print("Cupy not installed, using CPU only")
  

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

def calculate_frechet_distance_cupy(mu1, sigma1, mu2, sigma2, eps=1e-6):
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

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            #raise ValueError("Imaginary component {}".format(m))
            print("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean



def get_w2(act1,act2, device='cpu'):
    # act1 = batch_two[combination[0]]
    # act2 = batch_three[combination[1]]
    """Calculate w2 between two sets

    Args:
        act1: First set
        act2: Second set

    Returns:
        float: The FCD score
    """
    if device == 'cuda':
        act1 = cp.asarray(act1)
        act2 = cp.asarray(act2)
        mu1 = cp.mean(act1, axis=0)
        sigma1 = cp.cov(act1.T)

        mu2 = cp.mean(act2, axis=0)
        sigma2 = cp.cov(act2.T)
        fcd_score = calculate_frechet_distance_cupy(
            mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2
        )

    else:
        mu1 = np.mean(act1, axis=0)
        sigma1 = np.cov(act1.T)

        mu2 = np.mean(act2, axis=0)
        sigma2 = np.cov(act2.T)
        fcd_score = calculate_frechet_distance_cpu(
            mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2
        )
    # return (combination[0],combination[1],fcd_score)
    return (fcd_score)




import numpy as np

def get_w2(act1, act2, device='cpu'):
    mu1 = np.mean(act1, axis=0)
    sigma1 = np.cov(act1.T)

    mu2 = np.mean(act2, axis=0)
    sigma2 = np.cov(act2.T)

    fcd_score = calculate_frechet_distance(
        mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2
    )
    return fcd_score



