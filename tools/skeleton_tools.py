from tools.quaternion_tools import *
import scipy.ndimage.filters as filters

# Define a class to represent a skeleton
class Skeleton(object):
    def __init__(self, offset, kinematic_tree, device):
        """
        Initialize the Skeleton class.

        Parameters:
        offset (torch.Tensor): The offset vectors for each joint.
        kinematic_tree (list): The kinematic tree structure of the skeleton.
        device (torch.device): The device (CPU or GPU) to use.
        """
        self.device = device
        self._raw_offset_np = offset.numpy()  # Store the offset as a NumPy array
        self._raw_offset = offset.clone().detach().to(device).float()  # Store the offset as a PyTorch tensor
        self._kinematic_tree = kinematic_tree  # Store the kinematic tree structure
        self._offset = None  # Placeholder for the computed offsets
        self._parents = [0] * len(self._raw_offset)  # Initialize parent indices for each joint
        self._parents[0] = -1  # The root joint has no parent
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]  # Set the parent for each joint in the chain

    def njoints(self):
        """
        Return the number of joints in the skeleton.
        """
        return len(self._raw_offset)

    def offset(self):
        """
        Return the computed offsets.
        """
        return self._offset

    def set_offset(self, offsets):
        """
        Set the computed offsets.

        Parameters:
        offsets (torch.Tensor): The new offsets to set.
        """
        self._offset = offsets.clone().detach().to(self.device).float()

    def kinematic_tree(self):
        """
        Return the kinematic tree structure.
        """
        return self._kinematic_tree

    def parents(self):
        """
        Return the parent indices for each joint.
        """
        return self._parents

    def get_offsets_joints_batch(self, joints):
        """
        Compute the offsets for a batch of joints.

        Parameters:
        joints (torch.Tensor): The joint positions with shape (batch_size, num_joints, 3).

        Returns:
        torch.Tensor: The computed offsets with shape (batch_size, num_joints, 3).
        """
        assert len(joints.shape) == 3
        _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[:, i] = torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None] * _offsets[:, i]

        self._offset = _offsets.detach()
        return _offsets

    def get_offsets_joints(self, joints):
        """
        Compute the offsets for a single set of joints.

        Parameters:
        joints (torch.Tensor): The joint positions with shape (num_joints, 3).

        Returns:
        torch.Tensor: The computed offsets with shape (num_joints, 3).
        """
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]

        self._offset = _offsets.detach()
        return _offsets

    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
        """
        Perform inverse kinematics on a batch of joints to compute quaternion parameters.

        Parameters:
        joints (numpy.ndarray): The joint positions with shape (num_frames, num_joints, 3).
        face_joint_idx (list): Indices of the joints used to determine the forward direction.
        smooth_forward (bool): Whether to smooth the forward direction using a Gaussian filter.

        Returns:
        numpy.ndarray: The quaternion parameters with shape (num_frames, num_joints, 4).
        """
        assert len(face_joint_idx) == 4
        '''Get Forward Direction'''
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]

        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')

        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        '''Get Root Rotation'''
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        '''Inverse Kinematics'''
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                u = self._raw_offset_np[chain[j+1]][np.newaxis,...].repeat(len(joints), axis=0)

                v = joints[:, chain[j+1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]

                rot_u_v = qbetween_np(u, v)

                R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:,chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params

    def forward_kinematics(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        """
        Perform forward kinematics to compute joint positions from quaternion parameters.

        Parameters:
        quat_params (torch.Tensor): The quaternion parameters with shape (batch_size, num_joints, 4).
        root_pos (torch.Tensor): The root joint positions with shape (batch_size, 3).
        skel_joints (torch.Tensor, optional): The skeleton joint positions with shape (batch_size, num_joints, 3).
        do_root_R (bool): Whether to apply the root rotation.

        Returns:
        torch.Tensor: The computed joint positions with shape (batch_size, num_joints, 3).
        """
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        joints = torch.zeros(quat_params.shape[:-1] + (3,)).to(self.device)
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(quat_params), -1).detach().to(self.device)
            for i in range(1, len(chain)):
                R = qmul(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot(R, offset_vec) + joints[:, chain[i-1]]
        return joints

    def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        """
        Perform forward kinematics to compute joint positions from quaternion parameters (NumPy version).

        Parameters:
        quat_params (numpy.ndarray): The quaternion parameters with shape (num_frames, num_joints, 4).
        root_pos (numpy.ndarray): The root joint positions with shape (num_frames, 3).
        skel_joints (numpy.ndarray, optional): The skeleton joint positions with shape (num_frames, num_joints, 3).
        do_root_R (bool): Whether to apply the root rotation.

        Returns:
        numpy.ndarray: The computed joint positions with shape (num_frames, num_joints, 3).
        """
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d_np(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        """
        Perform forward kinematics to compute joint positions from continuous 6D parameters (NumPy version).

        Parameters:
        cont6d_params (numpy.ndarray): The continuous 6D parameters with shape (num_frames, num_joints, 6).
        root_pos (numpy.ndarray): The root joint positions with shape (num_frames, 3).
        skel_joints (numpy.ndarray, optional): The skeleton joint positions with shape (num_frames, num_joints, 3).
        do_root_R (bool): Whether to apply the root rotation.

        Returns:
        numpy.ndarray: The computed joint positions with shape (num_frames, num_joints, 3).
        """
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(cont6d_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix_np(cont6d_params[:, 0])
            else:
                matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
            for i in range(1, len(chain)):
                matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]][..., np.newaxis]
                joints[:, chain[i]] = np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints

    def forward_kinematics_cont6d(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        """
        Perform forward kinematics to compute joint positions from continuous 6D parameters.

        Parameters:
        cont6d_params (torch.Tensor): The continuous 6D parameters with shape (batch_size, num_joints, 6).
        root_pos (torch.Tensor): The root joint positions with shape (batch_size, 3).
        skel_joints (torch.Tensor, optional): The skeleton joint positions with shape (batch_size, num_joints, 3).
        do_root_R (bool): Whether to apply the root rotation.

        Returns:
        torch.Tensor: The computed joint positions with shape (batch_size, num_joints, 3).
        """
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
        joints[..., 0, :] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])
            else:
                matR = torch.eye(3).expand((len(cont6d_params), -1, -1)).detach().to(cont6d_params.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]].unsqueeze(-1)
                joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints