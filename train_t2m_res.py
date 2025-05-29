# Import necessary libraries
import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.masktransformer.mask_transformer import ResidualTransformer
from models.masktransformer.mask_transformer_trainer import ResidualTransformerTrainer
from models.vectorquantizer.rvqvae_model import RVQVAE

from configs.train_config import TrainT2MOptions

from utils.pose_plotter import plot_3d_motion
from utils.skeleton_feature_extraction import recover_from_ric
from utils.opt_setup import get_opt
from utils.seed_setup import fixseed
from utils.skeleton_params import t2m_kinematic_chain, kit_kinematic_chain

from dataset.t2m_data_preparation import Text2MotionDataset
from motions.eval_data_loader import get_eval_data_loader
from models.text_motion_evaluator import EvaluatorModelWrapper


# Function to plot text-to-motion data
def plot_t2m(data, save_dir, captions, m_lengths):
    # Inverse transform the data to get the original motion data
    data = train_dataset.inv_transform(data)

    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        # Extract the motion data up to the specified length
        joint_data = joint_data[:m_lengths[i]]
        # Recover the joint data from the RIC representation
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        # Define the save path for the animation
        save_path = pjoin(save_dir, '%02d.mp4' % i)
        # Plot and save the 3D motion
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=fps, radius=radius)

# Function to load the VQ-VAE model
def load_vq_model():
    # Load the VQ-VAE model options from file
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    # Initialize the VQ-VAE model with the given options
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    # Load the model checkpoint
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {opt.vq_name}')
    vq_model.to(opt.device)
    return vq_model, vq_opt

# Main execution block
if __name__ == '__main__':
    # Parse command-line arguments
    parser = TrainT2MOptions()
    opt = parser.parse()
    # Set random seed for reproducibility
    fixseed(opt.seed)

    # Set device (CPU or GPU)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    # Define directories for saving models, animations, and logs
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/res/', opt.dataset_name, opt.name)

    # Create directories if they don't exist
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # Set dataset-specific parameters
    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    elif opt.dataset_name == 'kit': #TODO
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_len = 55
        kinematic_chain = kit_kinematic_chain
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'

    else:
        raise KeyError('The dataset is missing.')

    # Set the directory for text data
    opt.text_dir = pjoin(opt.data_root, 'texts')

    # Load the VQ-VAE model
    vq_model, vq_opt = load_vq_model()

    # Set the CLIP model version
    clip_version = 'ViT-B/32'

    # Update the number of tokens and quantizers based on the VQ-VAE model
    opt.num_tokens = vq_opt.nb_code
    opt.num_quantizers = vq_opt.num_quantizers

    # Initialize the ResidualTransformer model with the given options
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                          cond_mode='text',
                                          latent_dim=opt.latent_dim,
                                          ff_size=opt.ff_size,
                                          num_layers=opt.n_layers,
                                          num_heads=opt.n_heads,
                                          dropout=opt.dropout,
                                          clip_dim=512,
                                          shared_codebook=vq_opt.shared_codebook,
                                          cond_drop_prob=opt.cond_drop_prob,
                                            share_weight=opt.share_weight,
                                          clip_version=clip_version,
                                          opt=opt)

    # Calculate the total number of parameters in the model
    all_params = 0
    pc_transformer = sum(param.numel() for param in res_transformer.parameters_wo_clip())

    print(res_transformer)
    all_params += pc_transformer

    print('Total parameters: {:.2f}M'.format(all_params / 1000_000))

    # Load mean and standard deviation for normalization
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    # Define the paths for the training and validation split files
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    # Initialize the training and validation datasets
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)

    # Create data loaders for the training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # Get the evaluation data loader
    eval_val_loader, _ = get_eval_data_loader(dataset_opt_path, 32, 'val', device=opt.device)

    # Load the evaluation wrapper
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # Initialize the trainer
    trainer = ResidualTransformerTrainer(opt, res_transformer, vq_model)

    # Start training the model
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper, plot_eval=plot_t2m)