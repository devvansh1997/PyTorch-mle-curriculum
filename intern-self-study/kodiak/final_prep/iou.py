def iou(box_a, box_b) -> float:

    # get individual coordinates
    a_x1, a_y1, a_x2, a_y2 = box_a
    b_x1, b_y1, b_x2, b_y2 = box_b
    # a_x1, a_y1, a_x2, a_y2 = box_a.unbind(dim=-1)
    # b_x1, b_y1, b_x2, b_y2 = box_b.unbind(dim=-1)

    # get overlap coordinates
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    # intersection area calc
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # union area calc
    a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    union_area = max(1e-6, (a_area + b_area - inter_area))

    # iou calc
    iou = inter_area / union_area

    return iou


print(iou((0, 0, 2, 2), (1, 1, 3, 3)))
print(iou((0, 0, 2, 2), (3, 3, 5, 5)))
print(iou((0, 0, 2, 2), (0, 0, 2, 2)))
print(iou((0, 0, 2, 2), (5, 0, 7, 2)))
