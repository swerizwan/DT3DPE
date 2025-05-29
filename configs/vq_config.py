import argparse
import os
import torch

def arg_parse(is_train=False):
    """
    Parses command-line arguments and sets up the training or evaluation environment.
    
    :param is_train: Boolean flag indicating whether the script is in training mode.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset and training parameters
    parser.add_argument('--dataset_name', type=str, default='humanml3d', 
                        help='Name of the dataset directory')
    parser.add_argument("--gpu_id", type=int, default=0, 
                        help='ID of the GPU to use')
    parser.add_argument('--batch_size', default=256, type=int, 
                        help='Batch size for training')
    parser.add_argument('--window_size', type=int, default=64, 
                        help='Length of motion sequences used during training')
    # Training schedule and learning rate
    parser.add_argument('--max_epoch', default=50, type=int, 
                        help='Total number of epochs to run')
    parser.add_argument('--lr', default=2e-4, type=float, 
                        help='Maximum learning rate')
    parser.add_argument('--warm_up_iter', default=2000, type=int, 
                        help='Number of warm-up iterations')
    parser.add_argument('--milestones', default=[150000, 250000], nargs="+", type=int, 
                        help="Learning rate schedule milestones (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help="Learning rate decay factor")

    # Regularization and loss functions
    parser.add_argument('--weight_decay', default=0.0, type=float, 
                        help='Weight decay (L2 regularization)')
    parser.add_argument("--commit", type=float, default=0.02, 
                        help="Commitment loss coefficient")
    parser.add_argument('--loss_vel', type=float, default=0.5, 
                        help='Velocity loss coefficient')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', 
                        help='Type of reconstruction loss')

    # Model architecture
    parser.add_argument("--code_dim", type=int, default=512, 
                        help="Dimension of the embedding space")
    parser.add_argument("--nb_code", type=int, default=512, 
                        help="Number of embeddings")
    parser.add_argument("--mu", type=float, default=0.99, 
                        help="Exponential moving average factor for codebook updates")
    parser.add_argument("--down_t", type=int, default=2, 
                        help="Temporal downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, 
                        help="Temporal stride size")
    parser.add_argument("--width", type=int, default=512, 
                        help="Width of the network")
    parser.add_argument("--depth", type=int, default=3, 
                        help="Number of residual blocks")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, 
                        help="Dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=512, 
                        help="Output embedding width")
    parser.add_argument('--vq_act', type=str, default='relu', choices=['relu', 'silu', 'gelu'],
                        help='Activation function for VQ layers')
    parser.add_argument('--vq_norm', type=str, default=None, 
                        help='Normalization type for VQ layers')

    # Quantization parameters
    parser.add_argument('--num_quantizers', type=int, default=3, 
                        help='Number of quantizers')
    parser.add_argument('--shared_codebook', action="store_true", 
                        help='Whether to share the codebook across quantizers')
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.2, 
                        help='Quantization dropout probability')

    # Experiment setup
    parser.add_argument('--ouput', type=str, default='default', 
                        help='Extension for the experiment')
    parser.add_argument('--name', type=str, default="test", 
                        help='Name of this trial')
    parser.add_argument('--is_continue', action="store_true", 
                        help='Whether to continue training from a previous checkpoint')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', 
                        help='Directory where models are saved')
    parser.add_argument('--log_every', default=10, type=int, 
                        help='Frequency of logging training progress (iterations)')
    parser.add_argument('--save_latest', default=500, type=int, 
                        help='Frequency of saving the latest model (iterations)')
    parser.add_argument('--save_every_e', default=2, type=int, 
                        help='Frequency of saving the model (epochs)')
    parser.add_argument('--eval_every_e', default=1, type=int, 
                        help='Frequency of evaluating the model (epochs)')
    parser.add_argument('--feat_bias', type=float, default=5, 
                        help='Bias for feature layers')

    # Evaluation parameters
    parser.add_argument('--which_epoch', type=str, default="all", 
                        help='Which epoch to evaluate')
    parser.add_argument('--vq_name', type=str, default="rvq_nq6_dc512_nc512_noshare_qdp0.2", 
                        help='Name of the VQ model')
    parser.add_argument("--seed", default=3407, type=int, 
                        help="Random seed for reproducibility")

    # Parse arguments
    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)

    # Print and save options
    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    # Set training flag
    opt.is_train = is_train

    # Save options to file if in training mode
    if is_train:
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

    return opt