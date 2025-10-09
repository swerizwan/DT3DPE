import os
import sys
import numpy as np
import torch
import argparse

from os.path import join as pjoin

from animation.IKSolvers import JacobianInverseKinematics, BasicInverseKinematics


# Define a softmax function with optional softness parameter
def softmax(x, **kw):
    """
    Computes the softmax of a given array.
    :param x: The input array.
    :param kw: Additional keyword arguments.
    :return: The softmax of the input array.
    """
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


# Define a softmin function using the softmax function
def softmin(x, **kw):
    """
    Computes the softmin of a given array.
    :param x: The input array.
    :param kw: Additional keyword arguments.
    :return: The softmin of the input array.
    """
    return -softmax(-x, **kw)


# Define a cubic alpha function for interpolation
def alpha(t):
    """
    Computes the cubic alpha function for interpolation.
    :param t: The input value.
    :return: The result of the cubic alpha function.
    """
    return 2.0 * t * t * t - 3.0 * t * t + 1


# Define a linear interpolation function
def lerp(a, l, r):
    """
    Performs linear interpolation between two values.
    :param a: The interpolation factor.
    :param l: The left value.
    :param r: The right value.
    :return: The interpolated value.
    """
    return (1 - a) * l + a * r


# Define a function to remove foot sliding from animation data
def remove_fs_old(anim, glb, foot_contact, fid_l=(3, 4), fid_r=(7, 8), interp_length=5, force_on_floor=True):
    """
    Removes foot sliding from animation data.
    :param anim: The animation data.
    :param glb: The global positions of the joints.
    :param foot_contact: The foot contact data.
    :param fid_l: The indices of the left foot joints.
    :param fid_r: The indices of the right foot joints.
    :param interp_length: The length of the interpolation window.
    :param force_on_floor: Whether to force the feet to be on the floor.
    :return: The modified animation data.
    """
    scale = 1. 
    height_thres = [0.06, 0.03] 

    if foot_contact is None:
        def foot_detect(positions, velfactor, heightfactor):
            """
            Detects foot contact based on velocity and height thresholds.
            :param positions: The joint positions.
            :param velfactor: The velocity threshold factor.
            :param heightfactor: The height threshold factor.
            :return: The detected foot contact data.
            """
            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            feet_l_h = positions[:-1, fid_l, 1]
            feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)

            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            feet_r_h = positions[:-1, fid_r, 1]

            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)

            return feet_l, feet_r

        feet_vel_thre = np.array([0.05, 0.2])
        feet_h_thre = np.array(height_thres) * scale
        feet_l, feet_r = foot_detect(glb, velfactor=feet_vel_thre, heightfactor=feet_h_thre)
        foot = np.concatenate([feet_l, feet_r], axis=-1).transpose(1, 0) 
        foot = np.concatenate([foot, foot[:, -1:]], axis=-1)
    else:
        foot = foot_contact.transpose(1, 0)

    T = len(glb)

    fid = list(fid_l) + list(fid_r)
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)  
    sort_height = np.sort(foot_heights)
    temp_len = len(sort_height)
    floor_height = np.mean(sort_height[int(0.25*temp_len):int(0.5*temp_len)])
    if floor_height > 0.5: 
        floor_height = 0
    glb[:, :, 1] -= floor_height
    anim.positions[:, 0, 1] -= floor_height
    for i, fidx in enumerate(fid):
        fixed = foot[i]  

        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            if force_on_floor:
                avg[1] = 0.0

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()

            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    targetmap = {}
    for j in range(glb.shape[1]):
        targetmap[j] = glb[:, j]
    ik = JacobianInverseKinematics(anim, targetmap, iterations=30, damping=5, recalculate=False, silent=True)

    anim = ik()
    return anim


# Define a function to remove foot sliding from global positions
def remove_fs(glb, foot_contact, fid_l=(3, 4), fid_r=(7, 8), interp_length=5, force_on_floor=True):
    """
    Removes foot sliding from global positions.
    :param glb: The global positions of the joints.
    :param foot_contact: The foot contact data.
    :param fid_l: The indices of the left foot joints.
    :param fid_r: The indices of the right foot joints.
    :param interp_length: The length of the interpolation window.
    :param force_on_floor: Whether to force the feet to be on the floor.
    :return: The modified global positions.
    """
    scale = 1. 
    height_thres = [0.06, 0.03] 
    if foot_contact is None:
        def foot_detect(positions, velfactor, heightfactor):
            """
            Detects foot contact based on velocity and height thresholds.
            :param positions: The joint positions.
            :param velfactor: The velocity threshold factor.
            :param heightfactor: The height threshold factor.
            :return: The detected foot contact data.
            """
            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            feet_l_h = positions[:-1, fid_l, 1]
            feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)

            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            feet_r_h = positions[:-1, fid_r, 1]

            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)

            return feet_l, feet_r

        feet_vel_thre = np.array([0.05, 0.2])
        feet_h_thre = np.array(height_thres) * scale
        feet_l, feet_r = foot_detect(glb, velfactor=feet_vel_thre, heightfactor=feet_h_thre)
        foot = np.concatenate([feet_l, feet_r], axis=-1).transpose(1, 0)  
        foot = np.concatenate([foot, foot[:, -1:]], axis=-1)
    else:
        foot = foot_contact.transpose(1, 0)

    T = len(glb)

    fid = list(fid_l) + list(fid_r)
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)  
    sort_height = np.sort(foot_heights)
    temp_len = len(sort_height)
    floor_height = np.mean(sort_height[int(0.25*temp_len):int(0.5*temp_len)])
    if floor_height > 0.5: 
        floor_height = 0
    glb[:, :, 1] -= floor_height
    for i, fidx in enumerate(fid):
        fixed = foot[i]  

        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            if force_on_floor:
                avg[1] = 0.0

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()

            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    targetmap = {}
    for j in range(glb.shape[1]):
        targetmap[j] = glb[:, j]

    return glb


# Define a function to compute foot sliding statistics
def compute_foot_sliding(foot_data, traj_qpos, offseth):
    """
    Computes foot sliding statistics.
    :param foot_data: The foot data.
    :param traj_qpos: The trajectory positions.
    :param offseth: The offset height.
    :return: The foot sliding statistics.
    """
    foot = np.array(foot_data).copy()
    offseth = np.mean(foot[:10, 1])
    foot[:, 1] -= offseth  
    foot_disp = np.linalg.norm(foot[1:, [0, 2]] - foot[:-1, [0, 2]], axis=1)
    traj_qpos[:, 1] -= offseth
    seq_len = len(traj_qpos)
    H = 0.05
    y_threshold = 0.65 
    y = traj_qpos[1:, 1]

    foot_avg = (foot[:-1, 1] + foot[1:, 1]) / 2
    subset = np.logical_and(foot_avg < H, y > y_threshold)

    sliding_stats = np.abs(foot_disp * (2 - 2 ** (foot_avg / H)))[subset]
    sliding = np.sum(sliding_stats) / seq_len * 1000
    return sliding, sliding_stats