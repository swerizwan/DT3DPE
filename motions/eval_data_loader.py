from dataset.t2m_data_preparation import Text2MotionDatasetEval, collate_fn  # TODO
from utils.word_embedding import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from utils.opt_setup import get_opt

# Function to get the evaluation data loader
def get_eval_data_loader(opt_path, batch_size, fname, device):
    # Load the options from the specified path
    opt = get_opt(opt_path, device)

    # Check if the dataset is recognized
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        print('Loading dataset %s ...' % opt.dataset_name)

        # Load the mean and standard deviation for normalization
        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        # Initialize the word vectorizer for text embeddings
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        # Define the path to the split file (e.g., 'train.txt' or 'val.txt')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        # Initialize the evaluation dataset
        dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
        # Create the data loader for the evaluation dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    else:
        # Raise an error if the dataset is not recognized
        raise KeyError('Dataset not Recognized !!')

    # Print a message indicating that the dataset has been loaded
    print('Ground Truth Dataset Loading Completed!!!')
    # Return the data loader and the dataset
    return dataloader, dataset