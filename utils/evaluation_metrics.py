import numpy as np
from scipy import linalg
import torch

# Function to calculate the Mean Per Joint Position Error (MPJPE)
def calculate_mpjpe(gt_joints, pred_joints):
    """
    Calculate the Mean Per Joint Position Error (MPJPE) between ground truth and predicted joints.

    Parameters:
    gt_joints (torch.Tensor): Ground truth joint positions with shape (num_poses, num_joints, 3).
    pred_joints (torch.Tensor): Predicted joint positions with shape (num_poses, num_joints, 3).

    Returns:
    torch.Tensor: MPJPE values for each pose.
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Center the joints around the pelvis (joint 0)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Calculate the Euclidean distance between predicted and ground truth joints
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1)
    mpjpe_seq = mpjpe.mean(-1)  # Average over joints for each pose

    return mpjpe_seq

# Function to calculate the Euclidean distance matrix between two sets of vectors
def euclidean_distance_matrix(matrix1, matrix2):
    """
    Calculate the Euclidean distance matrix between two sets of vectors.

    Parameters:
    matrix1 (numpy.ndarray): First set of vectors with shape (N1, D).
    matrix2 (numpy.ndarray): Second set of vectors with shape (N2, D).

    Returns:
    numpy.ndarray: Distance matrix with shape (N1, N2).
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    
    d3 = np.sum(np.square(matrix2), axis=1)    
    dists = np.sqrt(d1 + d2 + d3)  
    return dists

# Function to calculate the top-k accuracy
def calculate_top_k(mat, top_k):
    """
    Calculate the top-k accuracy for a given matrix.

    Parameters:
    mat (numpy.ndarray): Matrix with shape (N, K) where mat[i, j] is the rank of the j-th prediction for the i-th ground truth.
    top_k (int): Number of top predictions to consider.

    Returns:
    numpy.ndarray: Top-k accuracy matrix with shape (N, top_k).
    """
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat

# Function to calculate the R-precision
def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    """
    Calculate the R-precision between two sets of embeddings.

    Parameters:
    embedding1 (numpy.ndarray): First set of embeddings with shape (N1, D).
    embedding2 (numpy.ndarray): Second set of embeddings with shape (N2, D).
    top_k (int): Number of top predictions to consider.
    sum_all (bool): Whether to sum the results over all samples.

    Returns:
    numpy.ndarray: R-precision values.
    """
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat

# Function to calculate the matching score
def calculate_matching_score(embedding1, embedding2, sum_all=False):
    """
    Calculate the matching score between two sets of embeddings.

    Parameters:
    embedding1 (numpy.ndarray): First set of embeddings with shape (N1, D).
    embedding2 (numpy.ndarray): Second set of embeddings with shape (N2, D).
    sum_all (bool): Whether to sum the results over all samples.

    Returns:
    numpy.ndarray: Matching scores.
    """
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist

# Function to calculate the activation statistics
def calculate_activation_statistics(activations):
    """
    Calculate the mean and covariance of a set of activations.

    Parameters:
    activations (numpy.ndarray): Activations with shape (num_samples, dim_feat).

    Returns:
    tuple: Mean and covariance of the activations.
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

# Function to calculate the diversity of a set of activations
def calculate_diversity(activation, diversity_times):
    """
    Calculate the diversity of a set of activations.

    Parameters:
    activation (numpy.ndarray): Activations with shape (num_samples, dim_feat).
    diversity_times (int): Number of times to sample for diversity calculation.

    Returns:
    float: Diversity value.
    """
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

# Function to calculate the multimodality of a set of activations
def calculate_multimodality(activation, multimodality_times):
    """
    Calculate the multimodality of a set of activations.

    Parameters:
    activation (numpy.ndarray): Activations with shape (num_samples, num_per_sent, dim_feat).
    multimodality_times (int): Number of times to sample for multimodality calculation.

    Returns:
    float: Multimodality value.
    """
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_indices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_indices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_indices] - activation[:, second_indices], axis=2)
    return dist.mean()

# Function to calculate the Frechet distance
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate the Frechet distance between two multivariate Gaussians.

    Parameters:
    mu1 (numpy.ndarray): Mean of the first Gaussian.
    sigma1 (numpy.ndarray): Covariance of the first Gaussian.
    mu2 (numpy.ndarray): Mean of the second Gaussian.
    sigma2 (numpy.ndarray): Covariance of the second Gaussian.
    eps (float): Small value for numerical stability.

    Returns:
    float: Frechet distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)