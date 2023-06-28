import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_planes(num=12):
    if num == 1:
        return torch.tensor([[[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]], dtype=torch.float32)
    elif num == 3:
        return torch.tensor([[[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]],
                             [[1, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0]],
                             [[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]]], dtype=torch.float32)
    else:
        unormalized = torch.randn(num, 3, 3)
        Q = torch.qr(unormalized).Q
        return Q

#def project_onto_planes(planes, coordinates, perspective=False):
def project_onto_planes(planes, coordinates):
    # Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.inverse(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes.permute(0, 2, 1))

    plane_distances = projections[..., 2]
    plane_distances = plane_distances.reshape(N, n_planes, M)

    return projections[..., :2], plane_distances


def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear'):
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape

    plane_features = plane_features.view(N*n_planes, C, H, W)
    #projected_coordinates, plane_distances = project_onto_planes(plane_axes, coordinates, perspective=False)
    projected_coordinates, plane_distances = project_onto_planes(plane_axes, coordinates)
    projected_coordinates = projected_coordinates.unsqueeze(1)

    output_features = F.grid_sample(plane_features.float(), projected_coordinates.float(), mode='bilinear', align_corners=True, padding_mode='border').permute(0, 3, 2, 1).reshape(N, n_planes, M, C)

    return output_features, plane_distances

