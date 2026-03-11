import numpy as np
import torch
import random

# Function to fix the random seed for reproducibility
def fixseed(seed):
    """
    Fix the random seed for reproducibility.

    Parameters:
    seed (int): The seed value to use.
    """
    # Disable cuDNN benchmarking to ensure reproducibility
    torch.backends.cudnn.benchmark = False
    # Set the random seed for the random module
    random.seed(seed)
    # Set the random seed for NumPy
    np.random.seed(seed)
    # Set the random seed for PyTorch
    torch.manual_seed(seed)