# Import necessary libraries
import os
from os.path import join as pjoin

import torch
import torch.nn.functional as F

# Import custom models and utilities
from models.masktransformer.mask_transformer import MaskTransformer, ResidualTransformer
from models.vectorquantizer.rvqvae_model import RVQVAE, LengthEstimator

from configs.eval_config import EvalT2MOptions
from utils.opt_setup import get_opt

from utils.seed_setup import fixseed
from animation.BVHConverter import Joint2BVHConvertor
from torch.distributions.categorical import Categorical

from utils.skeleton_feature_extraction import recover_from_ric
from utils.pose_plotter import plot_3d_motion

from utils.skeleton_params import t2m_kinematic_chain

import numpy as np
clip_version = 'ViT-B/32'

# Function to load the VQ-VAE model
def load_vq_model(vq_opt):
    # Initialize the VQ-VAE model with the given options
    vq_model = RVQVAE(vq_opt,
                vq_opt.dim_pose,
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
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

# Function to load the T2M transformer model
def load_trans_model(model_opt, opt, which_model):
    # Initialize the T2M transformer model with the given options
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    # Load the model checkpoint
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

# Function to load the residual model
def load_res_model(res_opt, vq_opt, opt):
    # Update residual model options based on VQ-VAE model
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    # Initialize the residual model with the given options
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    # Load the model checkpoint
    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

# Function to load the length estimator model
def load_len_estimator(opt):
    # Initialize the length estimator model
    model = LengthEstimator(512, 50)
    # Load the model checkpoint
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location=opt.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model

# Main execution block
if __name__ == '__main__':
    # Parse command-line arguments
    parser = EvalT2MOptions()
    opt = parser.parse()
    # Set random seed for reproducibility
    fixseed(opt.seed)

    # Set device (CPU or GPU)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    # Determine the dimension of the pose data based on the dataset
    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    # Define directories for checkpoints, results, and animations
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./outputs', opt.ouput)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    # Create directories if they don't exist
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir, exist_ok=True)

    # Load model options from file
    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    # Load VQ-VAE model options and model
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    # Update model options based on VQ-VAE model
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    # Load residual model options and model
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt)

    # Ensure the VQ-VAE model name matches in both models
    assert res_opt.vq_name == model_opt.vq_name

    # Load T2M transformer model
    t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')

    # Load length estimator model
    length_estimator = load_len_estimator(model_opt)

    # Set models to evaluation mode
    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    length_estimator.eval()

    # Move models to the specified device
    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)
    length_estimator.to(opt.device)

    # Determine the number of joints based on the dataset
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    # Load mean and standard deviation for normalization
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))

    # Define inverse transformation function to denormalize data
    def inv_transform(data):
        return data * std + mean

    # Initialize lists for text prompts and motion lengths
    prompt_list = []
    length_list = []

    # Determine if motion lengths need to be estimated
    est_length = False
    if opt.text_prompt != "":
        prompt_list.append(opt.text_prompt)
        if opt.motion_length == 0:
            est_length = True
        else:
            length_list.append(opt.motion_length)
    elif opt.text_path != "":
        with open(opt.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                infos = line.split('#')
                prompt_list.append(infos[0])
                if len(infos) == 1 or (not infos[1].isdigit()):
                    est_length = True
                    length_list = []
                else:
                    length_list.append(int(infos[-1]))
    else:
        raise "You need to provide a text prompt or upload a file that contains text prompts!"

    # Estimate motion lengths if necessary
    if est_length:
        print("Since no motion lengths have been specified, we will use estimated values instead!")
        text_embedding = t2m_transformer.encode_text(prompt_list)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)  
        token_lens = Categorical(probs).sample() 
    else:
        token_lens = torch.LongTensor(length_list) // 4
        token_lens = token_lens.to(opt.device).long()

    # Update motion lengths based on token lengths
    m_length = token_lens * 4
    captions = prompt_list

    # Initialize sample counter and kinematic chain
    sample = 0
    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    # Repeat the generation process for the specified number of times
    for r in range(opt.repeat_times):
        print("-->Repeat %d" % r)
        with torch.no_grad():
            # Generate motions using the T2M transformer model
            mids = t2m_transformer.generate(captions, token_lens,
                                            timesteps=opt.time_steps,
                                            cond_scale=opt.cond_scale,
                                            temperature=opt.temperature,
                                            topk_filter_thres=opt.topkr,
                                            gsample=opt.gumbel_sample)
            # Refine generated motions using the residual model
            mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
            # Decode the generated tokens back to motion data
            pred_motions = vq_model.forward_decoder(mids)

            # Convert tensors to numpy arrays
            pred_motions = pred_motions.detach().cpu().numpy()

            # Denormalize the predicted motions
            data = inv_transform(pred_motions)

        # Save and visualize the generated motions
        for k, (caption, joint_data) in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d" % (k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir, str(k))

            # Create directories for each sample
            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            # Process joint data
            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

            # Convert joint data to BVH format and save
            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.bvh" % (k, r, m_length[k]))
            _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
            _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

            # Save animations as MP4 files
            save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4" % (k, r, m_length[k]))
            ik_save_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.mp4" % (k, r, m_length[k]))

            # Plot and save 3D motions
            plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
            plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy" % (k, r, m_length[k])), joint)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_ik.npy" % (k, r, m_length[k])), ik_joint)