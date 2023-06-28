import torch
from torch import nn
import numpy as np


def flatten_trajectories(data):
    # merge batch and trajectory dimensions in data dictionary
    for key in data.keys():
        if torch.is_tensor(data[key]):
            if data[key].ndim > 2:
                shape = [*data[key].shape]
                data[key] = data[key].reshape([shape[0] * shape[1]] + shape[2:])
    return data


def unflatten_trajectories(data, trajectory_length):
    # unmerge batch and trajectory dimensions in data dictionary
    for key in data.keys():
        if torch.is_tensor(data[key]):
            if data[key].ndim > 1:
                shape = [*data[key].shape]
                data[key] = data[key].reshape([-1, trajectory_length] + shape[1:])
    return data


def collapse_trajectory_dim(x):
    B, T = x.shape[:2]
    other_dims = x.shape[2:]
    return x.view(B * T, *other_dims)


def expand_trajectory_dim(x, T):
    other_dims = x.shape[1:]
    return x.view(-1, T, *other_dims)


def resize_trajectory(x, size):
    # Interpolation for image size, but for tensors with a trajectory dimension
    T = x.shape[1]
    x = collapse_trajectory_dim(x)
    x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)
    x = expand_trajectory_dim(x, T)
    return x


def ema_accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


class RenderParams:
    """Render parameters.

    A simple container for the variables required for rendering.
    """

    def __init__(self, Rt, K, samples_per_ray, near, far, alpha_noise_std=0, nerf_out_res=None, mask=None):
        self.samples_per_ray = samples_per_ray
        self.near = near
        self.far = far
        self.alpha_noise_std = alpha_noise_std
        self.Rt = Rt
        self.K = K
        self.mask = mask
        if nerf_out_res:
            self.nerf_out_res = nerf_out_res


