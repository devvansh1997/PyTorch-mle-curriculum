import torch
from torch import Tensor

def box_giou(boxes_a: Tensor, boxes_b: Tensor, eps: float = 1e-7) -> Tensor:
    
    # split our box tensors
    a_x1, a_y1, a_x2, a_y2 = boxes_a.unbind(dim=-1)
    b_x1, b_y1, b_x2, b_y2 = boxes_b.unbind(dim=-1)

    # get our overlap values (intersections)
    inter_x1 = torch.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = torch.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = torch.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = torch.minimum(a_y2[:, None], b_y2[None, :])

    # calc intersection area
    inter_width = (inter_x2 - inter_x1).clamp(min=0)
    inter_height = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_width * inter_height

    # calculate each box areas & union: A & B
    area_a = ((a_x2 - a_x1) * (a_y2 - a_y1))[:, None]
    area_b = ((b_x2 - b_x1) * (b_y2 - b_y1))[None, :]
    union = area_a + area_b - inter_area

    # iou calc
    iou = inter_area / (union + eps)

    # smalled enclosed box for GIoU
    # this one gets flipped
    enclosed_x1 = torch.minimum(a_x1[:, None], b_x1[None, :])
    enclosed_y1 = torch.minimum(a_y1[:, None], b_y1[None, :])
    enclosed_x2 = torch.maximum(a_x2[:, None], b_x2[None, :])
    enclosed_y2 = torch.maximum(a_y2[:, None], b_y2[None, :])
    enclosed_area = ((enclosed_x2 - enclosed_x1) * (enclosed_y2 - enclosed_y1))

    return iou - (enclosed_area - union) / (enclosed_area + eps)

example_01 = box_giou(Tensor([[0,0,10,10]]), Tensor([[0,0,10,10]]))
print(f"Example 01 - Full overlap: {example_01}")

example_02 = box_giou(Tensor([[0,0,10,10]]), Tensor([[10,0,20,10]]))
print(f"Example 02 - No overlap: {example_02}")

example_03 = box_giou(Tensor([[0,0,10,10]]), Tensor([[100,0,110,10]]))
print(f"Example 03 - Far apart on X-axis: {example_03}")

example_04 = box_giou(Tensor([[0,0,10,10]]), Tensor([[2,2,8,8]]))
print(f"Example 04 - Nested B inside A: {example_04}")