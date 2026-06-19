import torch
from torch import Tensor

def voxel_mean(
        points: Tensor,
        point_to_voxel: Tensor,
        num_voxels: int,
) -> Tensor:
    """
    returns: (M, 3) per voxel mean coordinates
    """
    M = num_voxels

    # counts points/voxel
    counts = torch.bincount(point_to_voxel, minlength=M)

    # sum points per voxel via scatter_add
    sums = torch.zeros((M, 3), dtype=points.dtype)
    index = point_to_voxel[:, None].expand(-1, 3)
    sums.scatter_add(dim=0, index=index, src=points)

    # divide for mean
    voxel_means = sums / counts[:, None].clamp(min=1)

    return voxel_means