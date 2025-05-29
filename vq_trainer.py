# Import necessary libraries
import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader

from models.vectorquantizer.rvqvae_model import RVQVAE
from models.vectorquantizer.rvq_trainer import RVQTokenizerTrainer
from configs.vq_config import arg_parse
from dataset.t2m_data_preparation import MotionDataset
from utils import paramUtil
import numpy as np

from models.text_motion_evaluator import EvaluatorModelWrapper
from utils.opt_setup import get_opt
from motions.eval_data_loader import get_eval_data_loader

from utils.skeleton_feature_extraction import recover_from_ric
from utils.pose_plotter import plot_3d_motion
from utils.seed_setup import fixseed

# Set the number of OpenMP threads to 1 to avoid issues with parallelism
os.environ["OMP_NUM_THREADS"] = "1"

# Function to plot text-to-motion data
def plot_t2m(data, save_dir):
    # Inverse transform the data to get the original motion data
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        # Recover the joint data from the RIC representation
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        # Define the save path for the animation
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        # Plot and save the 3D motion
        plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)

# Main execution block
if __name__ == "__main__":
    # Parse command-line arguments
    opt = arg_parse(True)
    # Set random seed for reproducibility
    fixseed(opt.seed)

    # Set the device (CPU or GPU)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    # Define directories for saving models, metadata, animations, and logs
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/vq/', opt.dataset_name, opt.name)

    # Create directories if they don't exist
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # Set dataset-specific parameters
    if opt.dataset_name == "t2m":
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        dim_pose = 263
        fps = 20
        radius = 4
        kinematic_chain = paramUtil.t2m_kinematic_chain
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    elif opt.dataset_name == "kit":
        opt.data_root = './dataset/KIT-ML/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
    else:
        raise KeyError('Dataset Does not Exists')

    # Load the dataset options and initialize the evaluation wrapper
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # Load mean and standard deviation for normalization
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    # Define the paths for the training and validation split files
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    # Initialize the RVQVAE model with the given options
    net = RVQVAE(opt,
                dim_pose,
                opt.nb_code,
                opt.code_dim,
                opt.code_dim,
                opt.down_t,
                opt.stride_t,
                opt.width,
                opt.depth,
                opt.dilation_growth_rate,
                opt.vq_act,
                opt.vq_norm)

    # Calculate the total number of parameters in the model
    pc_vq = sum(param.numel() for param in net.parameters())
    print(net)
    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))

    # Initialize the trainer
    trainer = RVQTokenizerTrainer(opt, vq_model=net)

    # Initialize the training and validation datasets
    train_dataset = MotionDataset(opt, mean, std, train_split_file)
    val_dataset = MotionDataset(opt, mean, std, val_split_file)

    # Create data loaders for the training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, pin_memory=True)
    eval_val_loader, _ = get_eval_data_loader(dataset_opt_path, 32, 'val', device=opt.device)

    # Start training the model
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper, plot_t2m)