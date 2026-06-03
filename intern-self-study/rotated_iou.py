import torch
from torch import Tensor

def box_to_corners(boxes: Tensor) -> Tensor:
    """
    box (N, 5): [cx, cy, l, w, theta] tensor
    returns: (N, 4, 2) tensor of (x,y) coordinates
    """

    # unbind box
    cx, cy, l, w, theta = boxes.unbind(dim=-1)

    # step 1: get corners of centered axis aligned box
    half_l, half_w = l/2, w/2
    x_local = torch.stack([half_l,  half_l, -half_l, -half_l], dim=-1)
    y_local = torch.stack([-half_w,  half_w,  half_w, -half_w], dim=-1)

    # step 2: rotate
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x_rotated = (x_local * cos_t[:, None]) - (y_local * sin_t[:, None])
    y_rotated = (x_local * sin_t[:, None]) + (y_local * cos_t[:, None])

    # step 3: translate
    x_world = x_rotated + cx[:, None]
    y_world = y_rotated + cy[:, None]

    return torch.stack([x_world, y_world], dim=-1)