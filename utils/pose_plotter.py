import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3

# Define a list of colors for plotting
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# Function to plot a 2D pose
def plot_2d_pose(pose, pose_tree, class_type, save_path=None, excluded_joints=None):
    """
    Plot a 2D pose.

    Parameters:
    pose (numpy.ndarray): The pose data with shape (num_joints, 2).
    pose_tree (list): The kinematic tree structure of the pose.
    class_type (str): The class type of the pose.
    save_path (str, optional): The path to save the plot.
    excluded_joints (list, optional): List of joints to exclude from the plot.
    """
    def init():
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(class_type)

    fig = plt.figure()
    init()
    data = np.array(pose, dtype=float)

    if excluded_joints is None:
        plt.scatter(data[:, 0], data[:, 1], color='b', marker='h', s=15)
    else:
        plot_joints = [i for i in range(data.shape[1]) if i not in excluded_joints]
        plt.scatter(data[plot_joints, 0], data[plot_joints, 1], color='b', marker='h', s=15)

    for idx1, idx2 in pose_tree:
        plt.plot([data[idx1, 0], data[idx2, 0]],
                 [data[idx1, 1], data[idx2, 1]], color='r', linewidth=2.0)

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()

# Function to average a list of values over intervals
def list_cut_average(ll, intervals):
    """
    Average a list of values over intervals.

    Parameters:
    ll (list): The list of values.
    intervals (int): The interval size.

    Returns:
    list: The averaged list.
    """
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new

# Function to plot a 3D pose
def plot_3d_pose_v2(savePath, kinematic_tree, joints, title=None):
    """
    Plot a 3D pose.

    Parameters:
    savePath (str): The path to save the plot.
    kinematic_tree (list): The kinematic tree structure of the pose.
    joints (numpy.ndarray): The joint positions with shape (num_joints, 3).
    title (str, optional): The title of the plot.
    """
    figure = plt.figure()
    ax = Axes3D(figure)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=110, azim=90)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='black')
    colors = ['red', 'magenta', 'black', 'magenta', 'black', 'green', 'blue']
    for chain, color in zip(kinematic_tree, colors):
        ax.plot3D(joints[chain, 0], joints[chain, 1], joints[chain, 2], linewidth=2.0, color=color)
    plt.show()

# Function to plot a 3D motion
def plot_3d_motion_v2(motion, kinematic_tree, save_path, interval=50, dataset=None, title=None):
    """
    Plot a 3D motion.

    Parameters:
    motion (numpy.ndarray): The motion data with shape (num_frames, num_joints, 3).
    kinematic_tree (list): The kinematic tree structure of the motion.
    save_path (str): The path to save the plot.
    interval (int, optional): The interval between frames.
    dataset (str, optional): The dataset name.
    title (str, optional): The title of the plot.
    """
    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim(0, 800)
        ax.set_xlim(0, 800)
        ax.set_zlim(0, 5000)
        if title is not None:
            ax.set_title(title)

    motion = motion.reshape(motion.shape[0], -1, 3)
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    init()

    data = np.array(motion, dtype=float)
    colors = ['red', 'magenta', 'black', 'green', 'blue','red', 'magenta', 'black', 'green', 'blue']
    frame_number = data.shape[0]
    print(data.shape)

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        ax.scatter(motion[index, :, 0], motion[index, :, 1], motion[index, :, 2], color='black')
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2], linewidth=2.0, color=color)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=True, repeat_delay=50)

    ani.save(save_path, writer='pillow')
    plt.close()

# Function to plot a 3D motion for the KIT dataset
def plot_3d_motion_kit(save_path, kinematic_tree, joints, title, figsize=(5, 5), interval=100, radius=246 * 12):
    """
    Plot a 3D motion for the KIT dataset.

    Parameters:
    save_path (str): The path to save the plot.
    kinematic_tree (list): The kinematic tree structure of the motion.
    joints (numpy.ndarray): The joint positions with shape (num_frames, num_joints, 3).
    title (str): The title of the plot.
    figsize (tuple, optional): The figure size.
    interval (int, optional): The interval between frames.
    radius (float, optional): The radius of the plot.
    """
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'magenta', 'black', 'green', 'blue', 'red', 'magenta', 'black', 'green', 'blue']
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='black')
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=2.0, color=color)
        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=True, repeat_delay=50)

    ani.save(save_path, writer='pillow')
    plt.close()

# Function to plot a 3D motion with ground truth and prediction
def plot_3d_motion_gt_pred(save_path, kinematic_tree, gt_joints, pred_joints, title, figsize=(10, 10), fps=120, radius=4):
    """
    Plot a 3D motion with ground truth and prediction.

    Parameters:
    save_path (str): The path to save the plot.
    kinematic_tree (list): The kinematic tree structure of the motion.
    gt_joints (numpy.ndarray): The ground truth joint positions with shape (num_frames, num_joints, 3).
    pred_joints (numpy.ndarray): The predicted joint positions with shape (num_frames, num_joints, 3).
    title (str): The title of the plot.
    figsize (tuple, optional): The figure size.
    fps (int, optional): The frames per second.
    radius (float, optional): The radius of the plot.
    """
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        for ax in axs:
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([0, radius])
            ax.set_zlim3d([0, radius])
            fig.suptitle(title, fontsize=20)
            ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz, ax):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    def update(index):
        for i, ax in enumerate(axs):
            ax.lines = []
            ax.collections = []
            ax.view_init(elev=120, azim=-90)
            ax.dist = 7.5

            MINS = motions_min[i]
            MAXS = motions_max[i]
            trajec = motions_traj[i]
            data = motions_data[i]

            plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                         MAXS[2] - trajec[index, 1], ax)

            if index > 1:
                ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                          trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                          color='blue')

            for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0
                ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                          color=color)

            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

    motions_data = []
    motions_traj = []
    motions_min = []
    motions_max = []
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    for i, joints in enumerate((gt_joints, pred_joints)):
        data = joints.copy().reshape(len(joints), -1, 3)
        frame_number = data.shape[0]

        MINS = data.min(axis=0).min(axis=0)
        motions_min.append(MINS)
        MAXS = data.max(axis=0).max(axis=0)
        motions_max.append(MAXS)
        height_offset = MINS[1]

        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]
        motions_traj.append(trajec)

        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]
        motions_data.append(data)

    axs = []
    fig = plt.figure(figsize=(20,10))
    axs.append(fig.add_subplot(1, 2, 1, projection='3d'))
    axs.append(fig.add_subplot(1, 2, 2, projection='3d'))
    init()

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    ani.save(save_path, fps=fps)
    plt.close()

# Function to plot a 3D motion
def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    """
    Plot a 3D motion.

    Parameters:
    save_path (str): The path to save the plot.
    kinematic_tree (list): The kinematic tree structure of the motion.
    joints (numpy.ndarray): The joint positions with shape (num_frames, num_joints, 3).
    title (str): The title of the plot.
    figsize (tuple, optional): The figure size.
    fps (int, optional): The frames per second.
    radius (float, optional): The radius of the plot.
    """
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    ani.save(save_path, fps=fps)
    plt.close()

# Function to plot a 3D motion (old version)
def plot_3d_motion_old(motion, pose_tree, class_type, save_path, interval=300, excluded_joints=None):
    """
    Plot a 3D motion (old version).

    Parameters:
    motion (numpy.ndarray): The motion data with shape (num_frames, num_joints, 3).
    pose_tree (list): The kinematic tree structure of the motion.
    class_type (str): The class type of the motion.
    save_path (str): The path to save the plot.
    interval (int, optional): The interval between frames.
    excluded_joints (list, optional): List of joints to exclude from the plot.
    """
    matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim(-0.75, 0.75)
        ax.set_xlim(-0.75, 0.75)
        ax.set_zlim(-0.75, 0.75)
        ax.set_title(class_type)

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    init()

    data = np.array(motion, dtype=float)
    frame_number = data.shape[0]
    print(data.shape)

    def update(index):
        ax.lines = []
        ax.collections = []
        if excluded_joints is None:
            ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='b', marker='h', s=15)
        else:
            plot_joints = [i for i in range(data.shape[1]) if i not in excluded_joints]
            ax.scatter(data[index, plot_joints, 0], data[index, plot_joints, 1], data[index, plot_joints, 2], color='b', marker='h', s=15)

        for idx1, idx2 in pose_tree:
            ax.plot([data[index, idx1, 0], data[index, idx2, 0]],
                    [data[index, idx1, 1], data[index, idx2, 1]], [data[index, idx1, 2], data[index, idx2, 2]], color='r', linewidth=2.0)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    ani.save(save_path, writer='pillow')
    plt.close()

# Function to plot a 2D motion
def plot_2d_motion(motion, pose_tree, axis_0, axis_1, class_type, save_path, interval=300):
    """
    Plot a 2D motion.

    Parameters:
    motion (numpy.ndarray): The motion data with shape (num_frames, num_joints, 3).
    pose_tree (list): The kinematic tree structure of the motion.
    axis_0 (int): The first axis to plot.
    axis_1 (int): The second axis to plot.
    class_type (str): The class type of the motion.
    save_path (str): The path to save the plot.
    interval (int, optional): The interval between frames.
    """
    matplotlib.use('Agg')

    fig = plt.figure()
    plt.title(class_type)
    data = np.array(motion, dtype=float)
    frame_number = data.shape[0]
    print(data.shape)

    def update(index):
        plt.clf()
        plt.xlim(-0.7, 0.7)
        plt.ylim(-0.7, 0.7)
        plt.scatter(data[index, :, axis_0], data[index, :, axis_1], color='b', marker='h', s=15)
        for idx1, idx2 in pose_tree:
            plt.plot([data[index, idx1, axis_0], data[index, idx2, axis_0]],
                     [data[index, idx1, axis_1], data[index, idx2, axis_1]], color='r', linewidth=2.0)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    ani.save(save_path, writer='pillow')
    plt.close()

# Function to plot multiple 3D motions
def plot_3d_multi_motion(motion_list, kinematic_tree, save_path, interval=50, dataset=None):
    """
    Plot multiple 3D motions.

    Parameters:
    motion_list (list): A list of motion data with shape (num_frames, num_joints, 3).
    kinematic_tree (list): The kinematic tree structure of the motion.
    save_path (str): The path to save the plot.
    interval (int, optional): The interval between frames.
    dataset (str, optional): The dataset name.
    """
    matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if dataset == "mocap":
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlim(0, 3)
            ax.set_zlim(-1.5, 1.5)
        else:
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    init()

    colors = ['red', 'magenta', 'black', 'magenta', 'black', 'green', 'blue']
    frame_number = motion_list[0].shape[0]
    print("Number of motions %d" % (len(motion_list)))

    def update(index):
        ax.lines = []
        ax.collections = []
        if dataset == "mocap":
            ax.view_init(elev=110, azim=-90)
        else:
            ax.view_init(elev=110, azim=90)
        for motion in motion_list:
            for chain, color in zip(kinematic_tree, colors):
                ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2],
                          linewidth=4.0, color=color)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    ani.save(save_path, writer='pillow')
    plt.close()