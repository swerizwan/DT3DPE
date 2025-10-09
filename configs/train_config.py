from configs.base_config import BaseOptions
import argparse

class TrainT2MOptions(BaseOptions):
    """
    Training options for Text-to-Motion (T2M) model.
    """
    def initialize(self):
        """
        Initializes the command-line arguments for training.
        """
        # Call the base class initialization
        BaseOptions.initialize(self)

        # Training parameters
        self.parser.add_argument('--batch_size', type=int, default=64, 
                                 help='Batch size for training')
        self.parser.add_argument('--max_epoch', type=int, default=500, 
                                 help='Maximum number of epochs for training')

        # Learning rate scheduler
        self.parser.add_argument('--lr', type=float, default=2e-4, 
                                 help='Initial learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, 
                                 help='Learning rate schedule factor')
        self.parser.add_argument('--milestones', default=[50_000], nargs="+", type=int,
                                 help="Learning rate schedule milestones (iterations)")
        self.parser.add_argument('--warm_up_iter', default=2000, type=int, 
                                 help='Number of warm-up iterations')

        # Condition and sampling
        self.parser.add_argument('--cond_drop_prob', type=float, default=0.1, 
                                 help='Drop ratio of condition for classifier-free guidance')
        self.parser.add_argument("--seed", default=3407, type=int, 
                                 help="Random seed for reproducibility")

        # Continuation and sampling strategy
        self.parser.add_argument('--is_continue', action="store_true", 
                                 help='Whether to continue training from a previous checkpoint')
        self.parser.add_argument('--gumbel_sample', action="store_true", 
                                 help='Sampling strategy: True for Gumbel sampling, False for categorical sampling')
        self.parser.add_argument('--share_weight', action="store_true", 
                                 help='Whether to share weights for projection/embedding in residual transformer')

        # Logging and evaluation
        self.parser.add_argument('--log_every', type=int, default=50, 
                                 help='Frequency of logging training progress (iterations)')
        self.parser.add_argument('--eval_every_e', type=int, default=10, 
                                 help='Frequency of evaluating the model (epochs)')
        self.parser.add_argument('--save_latest', type=int, default=500, 
                                 help='Frequency of saving the latest checkpoint (iterations)')

        # Set the training flag to True
        self.is_train = True


class TrainLenEstOptions():
    """
    Training options for Length Estimator model.
    """
    def __init__(self):
        """
        Initializes the argument parser for Length Estimator training.
        """
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--name', type=str, default="test", 
                                 help='Name of this trial')
        self.parser.add_argument("--gpu_id", type=int, default=-1, 
                                 help='GPU id to use')

        # Dataset and checkpoint directories
        self.parser.add_argument('--dataset_name', type=str, default='t2m', 
                                 help='Name of the dataset')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', 
                                 help='Directory where models are saved')

        # Training parameters
        self.parser.add_argument('--batch_size', type=int, default=64, 
                                 help='Batch size for training')
        self.parser.add_argument("--unit_length", type=int, default=4, 
                                 help="Length of motion units")
        self.parser.add_argument("--max_text_len", type=int, default=20, 
                                 help="Maximum length of text")

        self.parser.add_argument('--max_epoch', type=int, default=300, 
                                 help='Maximum number of epochs for training')
        self.parser.add_argument('--lr', type=float, default=1e-4, 
                                 help='Learning rate')

        # Continuation flag
        self.parser.add_argument('--is_continue', action="store_true", 
                                 help='Whether to continue training from a previous checkpoint')

        # Logging and evaluation frequencies
        self.parser.add_argument('--log_every', type=int, default=50, 
                                 help='Frequency of logging training progress')
        self.parser.add_argument('--save_every_e', type=int, default=5, 
                                 help='Frequency of saving checkpoints (epochs)')
        self.parser.add_argument('--eval_every_e', type=int, default=3, 
                                 help='Frequency of evaluation (epochs)')
        self.parser.add_argument('--save_latest', type=int, default=500, 
                                 help='Frequency of saving the latest checkpoint (iterations)')

    def parse(self):
        """
        Parses the command-line arguments and sets up the training configuration.
        """
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        return self.opt