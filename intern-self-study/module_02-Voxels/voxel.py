import torch
from torch import Tensor

def voxelize(
        points: Tensor,
        voxel_size: tuple[float, float, float],
        point_range: tuple[float, float, float, float, float, float]
) -> tuple[Tensor, Tensor]:
    """
    points: (N, 3) tensor containing LiDAR points in (x,y,z) format
    voxel_size: (vx, vy, vw) meters per voxel
    point_range: world bounds per axis (x_min, x_max, y_min, y_max, z _min, z_max)

    returns:
        voxel_indices: (M, 3) int64 - unique voxel (i, j, k) for each non-empty voxel
        point_to_voxel: (N',) int64 - for each kept point, the row in voxel_indices it belongs to. 
                         N' = the # of points the fell inside point_range  
    """
    # get boundaries
    x_min, y_min, z_min, x_max, y_max, z_max = point_range
    vx, vy, vz = voxel_size

    # filter for points outside the range
    range_min = torch.Tensor([x_min, y_min, z_min]) # (3,)
    range_max = torch.Tensor([x_max, y_max, z_max]) # (3,)
    in_range = ((points >= range_min) & (points < range_max)).all(dim=-1) # (N,)
    points = points[in_range] # (N', 3)

    # compute integer voxel index per point
    voxel_size_t = torch.Tensor([vx, vy, vz]) # (3,)
    voxel_ijk = ((points - range_min) / voxel_size_t).floor().long() # (N', 3)

    # now we will have some duplicates and we dont really need them so lets remove them
    voxel_indices, point_to_voxel = torch.unique(
        voxel_ijk, dim=0, return_inverse=True
    )   # voxel_indices : (M, 3) where M is the new number of valid voxels
        # point_to_voxel: (N') this is basically telling us which point maps to which voxel since we can have
        #                   more than one points/voxel

    return voxel_indices, point_to_voxel

print("EXAMPLE - 01")
points = torch.tensor([
    [ 0.10, 0.10, 0.10],
    [ 0.15, 0.12, 0.10],
    [ 0.25, 0.30, 0.40],
    [ 99.0, 0.0, 0.0],     # out of range -> filtered
])
voxel_size = (0.1, 0.1, 0.1)
point_range = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

voxel_indices, point_to_voxel = voxelize(points, voxel_size, point_range)
print("voxel_indices:", voxel_indices)
print("point_to_voxel:", point_to_voxel)

# expected:
#   voxel_indices: 2 unique rows -> (1,1,1) and (2,3,4) (order may flip)
#   point_to_voxel: length
print("\n\nEXAMPLE - 02")

points = torch.tensor([
    [-50.0, -50.0, -3.0],   # leftmost-bottom corner -> voxel (0, 0, 0)
    [  0.0,   0.0,  0.0],   # center -> voxel ((50/0.2), (50/0.2), (3/0.2)) = (250, 250, 15)
    [ 49.9,  49.9,  0.9],   # near top-right corner -> voxel (499, 499, 19)
])
voxel_size = (0.2, 0.2, 0.2)
point_range = (-50.0, -50.0, -3.0, 50.0, 50.0, 1.0)

voxel_indices, point_to_voxel = voxelize(points, voxel_size, point_range)
print("voxel_indices:", voxel_indices)
print("point_to_voxel:", point_to_voxel)

# expected:
#   voxel_indices contains [0,0,0], [250,250,15], [499,499,19]
#   point_to_voxel: length 3, all different (no duplicates)
# this confirms re-anchoring works for negative range_min

print("\n\nEXAMPLE - 03")
points = torch.tensor([
    [0.0, 0.0, 0.0],    # exactly at range_min -> kept, voxel (0,0,0)
    [1.0, 0.5, 0.5],    # exactly at range_max on x -> FILTERED (< is exclusive)
    [0.99, 0.99, 0.99], # just inside upper bound -> kept
])
voxel_size = (0.1, 0.1, 0.1)
point_range = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

voxel_indices, point_to_voxel = voxelize(points, voxel_size, point_range)
print("voxel_indices:", voxel_indices)
print("point_to_voxel:", point_to_voxel)

# expected:
#   2 voxels: (0,0,0) and (9,9,9)
#   point_to_voxel: length 2 (one point was filtered)