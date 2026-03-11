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

from utils.skeleton_feature_extraction import recover_from_ric
from utils.pose_plotter import plot_3d_motion

from utils.skeleton_params import t2m_kinematic_chain

import numpy as np

from t2m_animation_generator import load_vq_model, load_res_model, load_trans_model

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
    result_dir = pjoin('./editing', opt.ouput)
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

    # Set models to evaluation mode
    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()

    # Move models to the specified device
    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)

    # Define maximum motion length and load mean and standard deviation for normalization
    max_motion_length = 196
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))

    # Define inverse transformation function to denormalize data
    def inv_transform(data):
        return data * std + mean

    # Load and preprocess the source motion data
    motion = np.load(opt.source_motion)
    m_length = len(motion)
    motion = (motion - mean) / std
    if max_motion_length > m_length:
        motion = np.concatenate([motion, np.zeros((max_motion_length - m_length, motion.shape[1]))], axis=0)
    motion = torch.from_numpy(motion)[None].to(opt.device)

    # Initialize lists for text prompts and motion lengths
    prompt_list = []
    length_list = []
    if opt.motion_length == 0:
        opt.motion_length = m_length
        print("Using default motion length.")

    # Add text prompt and motion length to lists
    prompt_list.append(opt.text_prompt)
    length_list.append(opt.motion_length)
    if opt.text_prompt == "":
        raise "Using an empty text prompt."

    # Calculate token lengths based on motion lengths
    token_lens = torch.LongTensor(length_list) // 4
    token_lens = token_lens.to(opt.device).long()

    # Update motion lengths based on token lengths
    m_length = token_lens * 4
    captions = prompt_list
    print_captions = captions[0]

    # Parse and process edit slices
    _edit_slice = opt.mask_edit_section
    edit_slice = []
    for eds in _edit_slice:
        _start, _end = eds.split(',')
        _start = eval(_start)
        _end = eval(_end)
        edit_slice.append([_start, _end])

    # Initialize sample counter and kinematic chain
    sample = 0
    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    # Encode the motion data using the VQ-VAE model
    with torch.no_grad():
        tokens, features = vq_model.encode(motion)

    # Build editing mask to mark sections to be edited
    edit_mask = torch.zeros_like(tokens[..., 0])
    seq_len = tokens.shape[1]
    for _start, _end in edit_slice:
        if isinstance(_start, float):
            _start = int(_start * seq_len)
            _end = int(_end * seq_len)
        else:
            _start //= 4
            _end //= 4
        edit_mask[:, _start: _end] = 1
        print_captions = f'{print_captions} [{_start * 4 / 20.}s - {_end * 4 / 20.}s]'
    edit_mask = edit_mask.bool()

    # Repeat the editing process for the specified number of times
    for r in range(opt.repeat_times):
        print("-->Repeat %d" % r)
        with torch.no_grad():
            # Perform editing using the T2M transformer model
            mids = t2m_transformer.edit(
                captions, tokens[..., 0].clone(), m_length // 4,
                timesteps=opt.time_steps,
                cond_scale=opt.cond_scale,
                temperature=opt.temperature,
                topk_filter_thres=opt.topkr,
                gsample=opt.gumbel_sample,
                force_mask=opt.force_mask,
                edit_mask=edit_mask.clone(),
            )
            # Optionally use the residual model to refine the edited motions
            if opt.use_res_model:
                mids = res_model.generate(mids, captions, m_length // 4, temperature=1, cond_scale=2)
            else:
                mids.unsqueeze_(-1)

            # Decode the edited tokens back to motion data
            pred_motions = vq_model.forward_decoder(mids)

            # Convert tensors to numpy arrays
            pred_motions = pred_motions.detach().cpu().numpy()
            source_motions = motion.detach().cpu().numpy()

            # Denormalize the predicted and source motions
            data = inv_transform(pred_motions)
            source_data = inv_transform(source_motions)

        # Save and visualize the edited motions
        for k, (caption, joint_data, source_data) in enumerate(zip(captions, data, source_data)):
            print("---->Sample %d: %s %d" % (k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir, str(k))

            # Create directories for each sample
            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            # Process joint data and source data
            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

            source_data = source_data[:m_length[k]]
            soucre_joint = recover_from_ric(torch.from_numpy(source_data).float(), 22).numpy()

            # Convert joint data to BVH format and save
            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.bvh" % (k, r, m_length[k]))
            _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
            _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

            # Save animations as MP4 files
            save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4" % (k, r, m_length[k]))
            ik_save_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.mp4" % (k, r, m_length[k]))
            source_save_path = pjoin(animation_path, "sample%d_source_len%d.mp4" % (k, m_length[k]))

            # Plot and save 3D motions
            plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=print_captions, fps=20)
            plot_3d_motion(save_path, kinematic_chain, joint, title=print_captions, fps=20)
            plot_3d_motion(source_save_path, kinematic_chain, soucre_joint, title='None', fps=20)

            # Save joint data as numpy files
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy" % (k, r, m_length[k])), joint)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_ik.npy" % (k, r, m_length[k])), ik_joint)