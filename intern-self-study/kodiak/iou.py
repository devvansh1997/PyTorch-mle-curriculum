import torch
from torch import Tensor

def box_iou(boxes_a: Tensor, boxes_b: Tensor, eps: float = 1e-7) -> Tensor:
    # split into corner tensors for easier broadcasting
    a_x1, a_y1, a_x2, a_y2 = boxes_a.unbind(dim=-1)
    b_x1, b_y1, b_x2, b_y2 = boxes_b.unbind(dim=-1)

    # calculate intersections
    inter_x1 = torch.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = torch.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = torch.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = torch.minimum(a_y2[:, None], b_y2[None, :])

    # to calc area we need width and height
    inter_width = (inter_x2 - inter_x1).clamp(min=0)
    inter_height = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_width * inter_height

    # calculate |A| and |B| areas
    area_a = ((a_x2 - a_x1) * (a_y2 - a_y1))[:, None]
    area_b = ((b_x2 - b_x1) * (b_y2 - b_y1))[None, :]
    union = area_a + area_b - inter_area

    # calc IoU
    return inter_area / (union + eps)