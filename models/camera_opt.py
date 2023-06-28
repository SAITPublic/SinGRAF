import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as Rot


class CameraPoseCarrier(nn.Module):
    def __init__(self, trajectory_sampler=None, pose_matrices=None, lock_axes=(), lock_t=(), random_init_params={}, jitter=False, t_jitter=0.0, r_jitter=0.0, alpha_activation='relu'):
        super().__init__()
        print('CameraPoseCarrier')
        if random_init_params:
            pose_matrices = sample_camera_normal_xz_simple(num_samples=random_init_params['N'])
        elif trajectory_sampler is not None:
            pose_matrices = trajectory_sampler.real_Rts.clone().detach().reshape(-1,4,4)  # N x 4 x 4
            exit() # debugging

        self.alpha_activation = alpha_activation

        xyz = pose_matrices.inverse()[:3,3]
        self.jitter = jitter

        assert(pose_matrices is not None)

        self.pose_batch = CameraPoseBatch(pose_matrices=pose_matrices, lock_axes=lock_axes, lock_t=lock_t, jitter=jitter, t_jitter=t_jitter, r_jitter=r_jitter)

        self.pose_matrices = pose_matrices
        self.trajectory_sampler = trajectory_sampler
        self.num_poses = len(pose_matrices)

    def normalize(self):
        self.pose_batch.normalize_sin_cos()

    def get_pose_matrices(self, indices=None):
        if indices is not None:
            return self.pose_batch.get_pose_matrices(indices) # n_traj x 4 x 4
        return self.pose_batch.get_pose_matrices(torch.arange(self.num_poses).to(indices.device))

    def get_occupancy(self, generator, local_latents, trajectories):
        B = local_latents.shape[0]
        query_points = trajectories.unsqueeze(0).expand(B, -1, -1)  # B x n_traj x 3
        query_points = query_points.to(local_latents.device)

        B, n_trajectories, _ = query_points.shape

        if local_latents.dtype == torch.float16:
            query_points = query_points.half()

        # get occupancies for all trajectories
        with torch.no_grad():
            # z is tensor for shape [B, z_dim]
            occupancy = generator(local_latents=local_latents, xyz=query_points)

        # bin mode doesn't work great with softplus, so use ReLU anyway in that case
        if (self.alpha_activation == 'relu'):
            # anything negative is unoccupied
            occupancy = torch.nn.functional.relu(occupancy)
        elif self.alpha_activation == 'softplus':
            occupancy = torch.nn.functional.softplus(occupancy)

        occupancy = occupancy.squeeze(1)  # B x n_traj
        return occupancy

    def sample_trajectories(self, generator, local_latents, *args):
        """Return trajectories that best traverse a given scene.

        Input:
        -----
        generator: SceneGenerator
            Generator object to be evaluated for occupancy.
        local_latents: torch.Tensor
            Local latent codes of shape [B, local_z_dim, H, W] corresponding to the scenes that will be evaluted.

        Return:
        ------
        Rts: torch.Tensor
            Trajectories of camera extrinsic matrices of shape [B, seq_len, 4, 4].

        """
        num_poses = self.num_poses
        # randomly choose 1k trajectories to sample from
        n_subsamples = min(num_poses, 1000)
        subset_indices = torch.multinomial(
            torch.ones(num_poses), num_samples=n_subsamples, replacement=False
        )
        with torch.no_grad():
            real_Rts = self.get_pose_matrices(subset_indices)  # 1000 x 4 x 4
            # real_Rts = self.pose_matrices[subset_indices]
            trajectories = real_Rts.float().inverse()[:, :3, 3].contiguous()  # 1000 x 3
            occupancy = self.get_occupancy(generator=generator, local_latents=local_latents, trajectories=trajectories)  # B x 1000
            sample_weights = nn.functional.softmin(occupancy + 1e-8, dim=-1)  # B x 1000

            nans = torch.isnan(sample_weights)
            sample_weights[nans] = 1 / 1000
            selected_indices = torch.multinomial(sample_weights, num_samples=1, replacement=False).squeeze(1)  # B

        Rts = self.get_pose_matrices(subset_indices[selected_indices])  # 1000 x 4 x 4 --> B x 4 x 4

        Rts = Rts.to(local_latents.device).unsqueeze(1)  # B x 1 x 4 x 4
        return Rts


class CameraPoseBatch(nn.Module):
    def __init__(self, pose_matrices=None, lock_axes=(), lock_t=(), jitter=False, t_jitter=0.3, r_jitter=45., inverse_model=True):
        super().__init__()
        # pose_matrices N x 4 x 4
        if inverse_model:
            pose_matrices = pose_matrices.inverse()  # keeping inverse model

        N = len(pose_matrices)
        data = [ [] for _ in range(9) ]
        for pose_matrix in pose_matrices:
            t, zyx = self.pose_matrix_init(pose_matrix)
            data[0].append(t[0:1].float())
            data[1].append(t[1:2].float())
            data[2].append(t[2:3].float())
            data[3].append(torch.cos(zyx[0:1]).float())
            data[4].append(torch.sin(zyx[0:1]).float())
            data[5].append(torch.cos(zyx[1:2]).float())
            data[6].append(torch.sin(zyx[1:2]).float())
            data[7].append(torch.cos(zyx[2:3]).float())
            data[8].append(torch.sin(zyx[2:3]).float())

        self.zero = torch.nn.Parameter(torch.zeros(N), requires_grad=False)

        self.x = torch.nn.Parameter(torch.cat(data[0]), requires_grad=not 'x' in  lock_t)
        self.y = torch.nn.Parameter(torch.cat(data[1]), requires_grad=not 'y' in  lock_t)
        self.z = torch.nn.Parameter(torch.cat(data[2]), requires_grad=not 'z' in  lock_t)

        self.cos_z = torch.nn.Parameter(torch.cat(data[3]), requires_grad=not 'z' in lock_axes)
        self.sin_z = torch.nn.Parameter(torch.cat(data[4]), requires_grad=not 'z' in lock_axes)
        self.cos_y = torch.nn.Parameter(torch.cat(data[5]), requires_grad=not 'y' in lock_axes)
        self.sin_y = torch.nn.Parameter(torch.cat(data[6]), requires_grad=not 'y' in lock_axes)
        self.cos_x = torch.nn.Parameter(torch.cat(data[7]), requires_grad=not 'x' in lock_axes)
        self.sin_x = torch.nn.Parameter(torch.cat(data[8]), requires_grad=not 'x' in lock_axes)

        self.inverse = inverse_model

        self.jitter = jitter
        if jitter:
            self.jitter_t_std = [t_jitter * (not 'x' in lock_t), t_jitter* (not 'y' in lock_t) * 0.2, t_jitter* (not 'z' in lock_t)]
            self.jitter_r_std = [r_jitter*(not 'z' in lock_axes), r_jitter*(not 'y' in lock_axes), r_jitter*(not 'x' in lock_axes) * 0.2]

    @staticmethod
    def pose_matrix_init(pose_matrix):
        try: pose_matrix = pose_matrix.numpy()
        except: pass
        t = pose_matrix[:3,3]
        R = Rot.from_matrix(pose_matrix[:3,:3])
        zyx = R.as_euler('zyx',degrees=False)
        return torch.from_numpy(t), torch.from_numpy(zyx.copy())

    def get_pose_matrices(self, inds):
        comps = {
            'x': self.x[inds],
            'y': self.y[inds],
            'z': self.z[inds],
            'cos_z': self.cos_z[inds],
            'sin_z': self.sin_z[inds],
            'cos_y': self.cos_y[inds],
            'sin_y': self.sin_y[inds],
            'cos_x': self.cos_x[inds],
            'sin_x': self.sin_x[inds]
        }
        if self.inverse:
            return self.create_matrix_inverse(comps)
        else:
            return self.create_matrix(comps)

    def create_matrix_inverse(self, comps):
        S = len(comps['x'])
        R = self.create_rot_matrix(comps)  # S x 3 x 3
        t = torch.stack([comps['x'], comps['y'], comps['z']], dim=1).unsqueeze(-1)  # S x 3 x 1
        if self.jitter:
            R,t = self.jitter_R_t(R, t)
        R_T = R.permute(0,2,1)  # S x 3 x 3
        t_T = torch.bmm(-R_T, t)  # S x 3 x 1
        mat = torch.cat([R_T, t_T], dim=2)  # S x 3 x 4
        row4 = torch.tensor([[[0.,0.,0.,1.]]]).to(comps['x'].device).expand(S, -1, -1)  # 1 x 1 x 4 --> S x 1 x 4
        mat = torch.cat([mat, row4.detach()], dim=1)  # S x 4 x 4
        return mat

    def jitter_R_t(self, R, t):
        # R: N x 3 x 3
        # t: N x 3 x 1
        N = R.shape[0]
        rot_std = torch.tensor(self.jitter_r_std).unsqueeze(0)  # 1 x 3
        degree_noise = torch.randn(N, 3) * rot_std  # N x 3
        rot_noise = Rot.from_euler('zyx', degree_noise, degrees=True).as_matrix()  # N x 3 x 3
        R_new = torch.bmm(R, torch.from_numpy(rot_noise).to(R.device).detach().float())  # N x 3 x 3

        t_std = torch.tensor(self.jitter_t_std).unsqueeze(0)  # 1 x 3
        t_noise = torch.randn(N, 3) * t_std  # N x 3
        t_new = t + t_noise.to(t.device).unsqueeze(-1).detach()  # N x 3 x 1

        return R_new, t_new

    def create_matrix(self, comps):
        S = len(comps['x'])
        R = self.create_rot_matrix(comps)  # S x 3 x 3
        t = torch.stack([comps['x'], comps['y'], comps['z']], dim=1).unsqueeze(-1)  # S x 3 x 1
        mat = torch.cat([R, t], dim=2)  # S x 3 x 4
        row4 = torch.tensor([[[0.,0.,0.,1.]]]).to(comps['x'].device).expand(S, -1, -1)  # 1 x 1 x 4 --> S x 1 x 4
        mat = torch.cat([mat, row4], dim=1)  # S x 4 x 4
        return mat  # 4 x 4

    def create_rot_matrix(self, comps):
        z = self.create_matrix_z(comps)
        y = self.create_matrix_y(comps)
        x = self.create_matrix_x(comps)
        rotmat = torch.bmm(z, torch.bmm(y, x))  # S x 3 x 3
        return rotmat

    def create_matrix_z(self, comps):
        # compos are dictionary with components sliced from class parameters
        S = len(comps['x'])
        return torch.stack([
            torch.stack([comps['cos_z'], -comps['sin_z'], self.zero[:S]], dim=1),  # S x 3
            torch.stack([comps['sin_z'], comps['cos_z'], self.zero[:S]], dim=1),  # S x 3
            torch.tensor([[0,0,1]]).to(comps['x'].device).expand(S,-1)  # S x 3
        ], dim=1)  # S x 3 x 3

    def create_matrix_y(self, comps):
        # compos are dictionary with components sliced from class parameters
        S = len(comps['x'])
        return torch.stack([
            torch.stack([comps['cos_y'], self.zero[:S], comps['sin_y']], dim=1),  # S x 3
            torch.tensor([[0, 1, 0]]).to(comps['x'].device).expand(S, -1),  # S x 3
            torch.stack([-comps['sin_y'], self.zero[:S], comps['cos_y']], dim=1)  # S x 3
        ], dim=1)  # S x 3 x 3

    def create_matrix_x(self, comps):
        # compos are dictionary with components sliced from class parameters
        S = len(comps['x'])
        return torch.stack([
            torch.tensor([[1, 0, 0]]).to(comps['x'].device).expand(S, -1),  # S x 3
            torch.stack([self.zero[:S], comps['cos_x'], -comps['sin_x']], dim=1),  # S x 3
            torch.stack([self.zero[:S], comps['sin_x'], comps['cos_x']], dim=1)  # S x 3
        ], dim=1)  # S x 3 x 3

    def normalize_sin_cos(self):
        self.normalize_sin_cos_(self.sin_x, self.cos_x)
        self.normalize_sin_cos_(self.sin_y, self.cos_y)
        self.normalize_sin_cos_(self.sin_z, self.cos_z)

    @staticmethod
    def normalize_sin_cos_(s,c):
        scale = (s.data**2 + c.data**2).sqrt()
        s.data /= scale
        c.data /= scale


def sample_camera_normal_xz_simple(std=1.65, y_height=0.0, num_samples=1):
    # 1.65 empirically from replica run of GSN
    poses = []
    for n in range(num_samples):
        angle = np.random.randint(-180, 180)
        angle = np.deg2rad(angle)
        _rand_rot = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])  # y rot
        rand_rot = torch.eye(4)
        rand_rot[:3, :3] = torch.from_numpy(_rand_rot).float()
        rand_rot[0,3] = torch.randn(1) * std
        rand_rot[2,3] = torch.randn(1) * std
        rand_rot[1,3] = y_height
        poses.append(rand_rot.inverse())
    pose_mats = torch.stack(poses, dim=0)
    return pose_mats


def draw_scatter(filename, voxel_res, voxel_size, xyz):
    import matplotlib.pyplot as plt

    try: xyz = xyz.cpu().numpy()
    except: pass

    for i in range(len(xyz)):
        plt.scatter(xyz[i, :, 0], xyz[i, :, 2], c='blue', alpha=0.1, linewidth=3)
    plt.xlabel('X')
    plt.ylabel('Z')
    extents = voxel_res * voxel_size / 2
    plt.xlim(-extents, extents)
    plt.ylim(-extents, extents)
    plt.savefig(filename)
    plt.close()


