import animation.animproc as Animation

from animation.IKSolvers import BasicInverseKinematics, BasicJacobianIK, InverseKinematics
from animation.quaternions import Quaternions
import animation.BVHAnim_mod as BVH
from animation.foot_sliding_removal import *

from utils.pose_plotter import plot_3d_motion
from utils import paramUtil
from tools.skeleton_tools import Skeleton
import torch

from torch import nn
from animation.utils.quaternion_utils import ik_rot, between, fk, ik
from tqdm import tqdm


def get_grot(glb, parent, offset):
    root_quat = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(glb.shape[0], axis=0)[:, None]
    local_pos = glb[:, 1:] - glb[:, parent[1:]]
    norm_offset = offset[1:] / np.linalg.norm(offset[1:], axis=-1, keepdims=True)
    norm_lpos = local_pos / np.linalg.norm(local_pos, axis=-1, keepdims=True)
    grot = between(norm_offset, norm_lpos)
    grot = np.concatenate((root_quat, grot), axis=1)
    grot /= np.linalg.norm(grot, axis=-1, keepdims=True)
    return grot


class Joint2BVHConvertor:
    def __init__(self):
        self.template = BVH.load('./animation/data/template.bvh', need_quater=True)
        self.re_order = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]

        self.re_order_inv = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 18, 13, 15, 19, 16, 20, 17, 21]
        self.end_points = [4, 8, 13, 17, 21]

        self.template_offset = self.template.offsets.copy()
        self.parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

    def convert(self, positions, filename, iterations=10, foot_ik=True):
        '''
        Convert the SMPL joint positions to Mocap BVH
        :param positions: (N, 22, 3)
        :param filename: Save path for resulting BVH
        :param iterations: iterations for optimizing rotations, 10 is usually enough
        :param foot_ik: whether to enfore foot inverse kinematics, removing foot slide issue.
        :return:
        '''
        positions = positions[:, self.re_order]
        new_anim = self.template.copy()
        new_anim.rotations = Quaternions.id(positions.shape[:-1])
        new_anim.positions = new_anim.positions[0:1].repeat(positions.shape[0], axis=-0)
        new_anim.positions[:, 0] = positions[:, 0]

        if foot_ik:
            positions = remove_fs(positions, None, fid_l=(3, 4), fid_r=(7, 8), interp_length=5,
                                  force_on_floor=True)
        ik_solver = BasicInverseKinematics(new_anim, positions, iterations=iterations, silent=True)
        new_anim = ik_solver()

        glb = Animation.positions_global(new_anim)[:, self.re_order_inv]
        if filename is not None:
            BVH.save(filename, new_anim, names=new_anim.names, frametime=1 / 20, order='zyx', quater=True)
        return new_anim, glb

    def convert_sgd(self, positions, filename, iterations=100, foot_ik=True):
        '''
        Convert the SMPL joint positions to Mocap BVH

        :param positions: (N, 22, 3)
        :param filename: Save path for resulting BVH
        :param iterations: iterations for optimizing rotations, 10 is usually enough
        :param foot_ik: whether to enfore foot inverse kinematics, removing foot slide issue.
        :return:
        '''
        glb = positions[:, self.re_order]

        if foot_ik:
             glb = remove_fs(glb, None, fid_l=(3, 4), fid_r=(7, 8), interp_length=2,
                                 force_on_floor=True)
        new_anim = self.template.copy()
        new_anim.rotations = Quaternions.id(glb.shape[:-1])
        new_anim.positions = new_anim.positions[0:1].repeat(glb.shape[0], axis=-0)
        new_anim.positions[:, 0] = glb[:, 0]
        anim = new_anim.copy()

        rot = torch.tensor(anim.rotations.qs, dtype=torch.float)
        pos = torch.tensor(anim.positions[:, 0, :], dtype=torch.float)
        offset = torch.tensor(anim.offsets, dtype=torch.float)

        glb = torch.tensor(glb, dtype=torch.float)
        ik_solver = InverseKinematics(rot, pos, offset, anim.parents, glb)
        print('Fixing foot contact using IK...')
        for i in tqdm(range(iterations)):
            mse = ik_solver.step()
        rotations = ik_solver.rotations.detach().cpu()
        norm = torch.norm(rotations, dim=-1, keepdim=True)
        rotations /= norm

        anim.rotations = Quaternions(rotations.numpy())
        anim.rotations[:, self.end_points] = Quaternions.id((anim.rotations.shape[0], len(self.end_points)))
        anim.positions[:, 0, :] = ik_solver.position.detach().cpu().numpy()
        if filename is not None:
            BVH.save(filename, anim, names=new_anim.names, frametime=1 / 20, order='zyx', quater=True)
        glb = Animation.positions_global(anim)[:, self.re_order_inv]
        return anim, glb



if __name__ == "__main__":
    folder = '/home/abbas/DT3DPE/demo/cond4_topkr0.9_ts18_tau1.0_s1009'
    files = os.listdir(os.path.join(folder, 'joints'))
    files = [f for f in files if 'repeat' in f]
    converter = Joint2BVHConvertor()
    for f in tqdm(files):
        joints = np.load(os.path.join(folder, 'joints', f))
        converter.convert(joints, os.path.join(folder, 'ik_animations', f'ik_{f}'.replace('npy', 'mp4')), foot_ik=True)