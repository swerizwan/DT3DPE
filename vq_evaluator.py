# Import necessary libraries
import sys
import os
from os.path import join as pjoin

import torch
from models.vectorquantizer.rvqvae_model import RVQVAE
from configs.vq_config import arg_parse
from motions.eval_data_loader import get_eval_data_loader
import utils.t2m_model_evaluation as eval_t2m
from utils.opt_setup import get_opt
from models.text_motion_evaluator import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from utils.word_embedding import WordVectorizer

# Function to load the VQ-VAE model
def load_vq_model(vq_opt, which_epoch):
    # Initialize the VQ-VAE model with the given options
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.code_dim,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    # Load the model checkpoint
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', which_epoch),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    # Get the epoch number from the checkpoint
    vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
    print(f'Loading VQ Model {vq_opt.name} Completed!, Epoch {vq_epoch}')
    return vq_model, vq_epoch

# Main execution block
if __name__ == "__main__":
    # Parse command-line arguments
    args = arg_parse(False)
    # Set the device (CPU or GPU)
    args.device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))

    # Define the output directory for evaluation results
    args.out_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'eval')
    # Create the output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Open a log file for writing evaluation results
    f = open(pjoin(args.out_dir, '%s.log'%args.ext), 'w')

    # Define the dataset options path based on the dataset name
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataset_name == 'kit' \
                                                        else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    # Load the dataset options and initialize the evaluation wrapper
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    # Set the number of joints based on the dataset
    args.nb_joints = 21 if args.dataset_name == 'kit' else 22
    dim_pose = 251 if args.dataset_name == 'kit' else 263

    # Get the evaluation data loader
    eval_val_loader, _ = get_eval_data_loader(dataset_opt_path, 32, 'test', device=args.device)

    # Print the length of the evaluation data loader
    print(len(eval_val_loader))

    # Load the VQ-VAE model options
    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=args.device)
    # Define the model directory
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')

    # Iterate over the model checkpoints in the model directory
    for file in os.listdir(model_dir):
        # Skip files that do not match the specified epoch
        if args.which_epoch != "all" and args.which_epoch not in file:
            continue
        print(file)
        # Load the VQ-VAE model
        net, ep = load_vq_model(vq_opt, file)

        # Set the model to evaluation mode and move it to the GPU
        net.eval()
        net.cuda()

    # Close the log file
    f.close()
