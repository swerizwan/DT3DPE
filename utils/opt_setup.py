import os
from argparse import Namespace
import re
from os.path import join as pjoin
from utils.word_embedding import POS_enumerator

# Function to check if a string represents a float
def is_float(numStr):
    """
    Check if a string represents a float.

    Parameters:
    numStr (str): The string to check.

    Returns:
    bool: True if the string represents a float, False otherwise.
    """
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag

# Function to check if a string represents an integer
def is_number(numStr):
    """
    Check if a string represents an integer.

    Parameters:
    numStr (str): The string to check.

    Returns:
    bool: True if the string represents an integer, False otherwise.
    """
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')   
    if str(numStr).isdigit():
        flag = True
    return flag

# Function to load and parse options from a file
def get_opt(opt_path, device, **kwargs):
    """
    Load and parse options from a file.

    Parameters:
    opt_path (str): Path to the options file.
    device (torch.device): The device (CPU or GPU) to use.
    **kwargs: Additional keyword arguments to update the options.

    Returns:
    Namespace: The parsed options.
    """
    opt = Namespace()
    opt_dict = vars(opt)

    # Define lines to skip in the options file
    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path, 'r') as f:
        for line in f:
            if line.strip() not in skip:
                key, value = line.strip('\n').split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # Set default values for certain options
    opt_dict['which_epoch'] = 'finest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    # Set dataset-specific parameters
    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.dim_pose = 251
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    else:
        raise KeyError('Dataset not recognized')
    
    # Set additional default values for options
    if not hasattr(opt, 'unit_length'):
        opt.unit_length = 4
    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    # Update options with any additional keyword arguments
    opt_dict.update(kwargs)

    return opt