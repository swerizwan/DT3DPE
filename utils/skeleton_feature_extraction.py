from os.path import join as pjoin

from tools.skeleton_tools import Skeleton
import numpy as np
import os
from tools.quaternion_tools import *
from utils.skeleton_params import *

import torch

# Function to uniformize the skeleton
def uniform_skeleton(positions, target_offset):
    """
    Uniformize the skeleton to match the target offset.

    Parameters:
    positions (numpy.ndarray): The joint positions with shape (num_frames, num_joints, 3).
    target_offset (torch.Tensor): The target offset tensor.

    Returns:
    numpy.ndarray: The uniformized joint positions.
    """
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints

# Function to extract features from joint positions
def extract_features(positions, feet_thre, n_raw_offsets, kinematic_chain, face_joint_indx, fid_r, fid_l):
    """
    Extract features from joint positions.

    Parameters:
    positions (numpy.ndarray): The joint positions with shape (num_frames, num_joints, 3).
    feet_thre (float): The threshold for foot detection.
    n_raw_offsets (torch.Tensor): The raw offsets tensor.
    kinematic_chain (list): The kinematic chain structure.
    face_joint_indx (list): The indices of joints used to determine the forward direction.
    fid_r (list): The indices of the right foot joints.
    fid_l (list): The indices of the left foot joints.

    Returns:
    numpy.ndarray: The extracted features.
    """
    global_positions = positions.copy()
    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        r_rot = quat_params[:, 0].copy()
        '''Root Linear Velocity'''
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()
        '''Root Linear Velocity'''
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data

# Function to process a file and extract features
def process_file(positions, feet_thre):
    """
    Process a file and extract features.

    Parameters:
    positions (numpy.ndarray): The joint positions with shape (num_frames, num_joints, 3).
    feet_thre (float): The threshold for foot detection.

    Returns:
    tuple: The extracted features, global positions, local positions, and linear velocity.
    """
    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    '''New ground truth positions'''
    global_positions = positions.copy()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
        return feet_l, feet_r
    feet_l, feet_r = foot_detect(positions, feet_thre)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        r_rot = quat_params[:, 0].copy()
        '''Root Linear Velocity'''
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()
        '''Root Linear Velocity'''
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity

# Function to recover root rotation and position
def recover_root_rot_pos(data):
    """
    Recover root rotation and position from data.

    Parameters:
    data (torch.Tensor): The input data tensor.

    Returns:
    tuple: The recovered root rotation quaternion and root position.
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

# Function to recover joint positions from rotation data
def recover_from_rot(data, joints_num, skeleton):
    """
    Recover joint positions from rotation data.

    Parameters:
    data (torch.Tensor): The input data tensor.
    joints_num (int): The number of joints.
    skeleton (Skeleton): The skeleton object.

    Returns:
    torch.Tensor: The recovered joint positions.
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions

# Function to recover joint positions from rotation invariant coordinates
def recover_from_ric(data, joints_num):
    """
    Recover joint positions from rotation invariant coordinates.

    Parameters:
    data (torch.Tensor): The input data tensor.
    joints_num (int): The number of joints.

    Returns:
    torch.Tensor: The recovered joint positions.
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

if __name__ == "__main__":
    example_id = "03950_gt"
    l_idx1, l_idx2 = 17, 18
    fid_r, fid_l = [14, 15], [19, 20]
    face_joint_indx = [11, 16, 5, 8]
    r_hip, l_hip = 11, 16
    joints_num = 21
    data_dir = '../dataset/kit_mocap_dataset/joints/'
    save_dir1 = '../dataset/kit_mocap_dataset/new_joints/'
    save_dir2 = '../dataset/kit_mocap_dataset/new_joint_vecs/'

    n_raw_offsets = torch.from_numpy(kit_raw_offsets)
    kinematic_chain = kit_kinematic_chain

    '''Get offsets of target skeleton'''
    examples = np.load(os.path.join(data_dir, example_id + '.npy'))
    examples = examples.reshape(len(examples), -1, 3)
    examples = torch.from_numpy(examples)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(examples[0])

    source_list = os.listdir(data_dir)
    frame_num = 0
    '''Read source data'''
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(data_dir, source_file))[:, :joints_num]
        try:
            name = ''.join(source_file[:-7].split('_')) + '.npy'
            data, ground_positions, positions, l_velocity = process_file(source_data, 0.05)
            rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
            if np.isnan(rec_ric_data.numpy()).any():
                print(source_file)
                continue
            np.save(pjoin(save_dir1, name), rec_ric_data.squeeze().numpy())
            np.save(pjoin(save_dir2, name), data)
            frame_num += data.shape[0]
        except Exception as e:
            print(source_file)
            print(e)

    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 12.5 / 60))