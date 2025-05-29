# Import necessary libraries
import os
from os.path import join as pjoin

import torch

# Import custom models and utilities
from models.masktransformer.mask_transformer import MaskTransformer, ResidualTransformer
from models.vectorquantizer.rvqvae_model import RVQVAE

from configs.eval_config import EvalT2MOptions
from utils.opt_setup import get_opt
from motions.eval_data_loader import get_eval_data_loader
from models.text_motion_evaluator import EvaluatorModelWrapper

import utils.t2m_model_evaluation as eval_t2m
from utils.seed_setup import fixseed

import numpy as np

# Function to load the VQ-VAE model
def load_vq_model(vq_opt):
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
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

# Function to load the T2M transformer model
def load_trans_model(model_opt, which_model):
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
                      map_location=opt.device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

# Function to load the residual model
def load_res_model(res_opt):
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

    # Define directories for checkpoints and evaluation results
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    out_dir = pjoin(root_dir, 'eval')
    # Create the evaluation directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Define the output log file path
    out_path = pjoin(out_dir, "%s.log"%opt.output)

    # Open the log file for writing
    f = open(pjoin(out_path), 'w')

    # Load model options from file
    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    clip_version = 'ViT-B/32'

    # Load VQ-VAE model options and model
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    # Update model options based on VQ-VAE model
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    # Load residual model options and model
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt)

    # Ensure the VQ-VAE model name matches in both models
    assert res_opt.vq_name == model_opt.vq_name

    # Load dataset options and evaluation wrapper
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if opt.dataset_name == 'kit' \
        else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # Determine the number of joints based on the dataset
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    # Get the evaluation data loader
    eval_val_loader, _ = get_eval_data_loader(dataset_opt_path, 32, 'test', device=opt.device)

    # Iterate over the model checkpoints in the model directory
    for file in os.listdir(model_dir):
        if opt.which_epoch != "all" and opt.which_epoch not in file:
            continue
        print('loading checkpoint {}'.format(file))
        # Load the T2M transformer model
        t2m_transformer = load_trans_model(model_opt, file)
        t2m_transformer.eval()
        vq_model.eval()
        res_model.eval()

        # Move models to the specified device
        t2m_transformer.to(opt.device)
        vq_model.to(opt.device)
        res_model.to(opt.device)

        # Initialize lists to store evaluation metrics
        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        mm = []

        # Number of times to repeat the evaluation
        repeat_time = 20
        for i in range(repeat_time):
            with torch.no_grad():
                # Perform evaluation using the evaluation wrapper
                best_fid, best_div, Rprecision, best_matching, best_mm = \
                    eval_t2m.evaluation_mask_transformer_test_plus_res(eval_val_loader, vq_model, res_model, t2m_transformer,
                                                                       i, eval_wrapper=eval_wrapper,
                                                         time_steps=opt.time_steps, cond_scale=opt.cond_scale,
                                                         temperature=opt.temperature, topkr=opt.topkr,
                                                                       force_mask=opt.force_mask, cal_mm=True)
            # Append the evaluation metrics to the lists
            fid.append(best_fid)
            div.append(best_div)
            top1.append(Rprecision[0])
            top2.append(Rprecision[1])
            top3.append(Rprecision[2])
            matching.append(best_matching)
            mm.append(best_mm)

        # Convert lists to numpy arrays for easier manipulation
        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        mm = np.array(mm)

        # Print and log the final evaluation results
        print(f'{file} final result:')
        print(f'{file} final result:', file=f, flush=True)

        msg_final = f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
        print(msg_final)
        print(msg_final, file=f, flush=True)

    # Close the log file
    f.close()