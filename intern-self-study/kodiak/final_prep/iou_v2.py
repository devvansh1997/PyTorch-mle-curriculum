import numpy as np


def iou(boxes_a, boxes_b):
    # extract coordinates
    a_x1 = boxes_a[:, 0]
    a_y1 = boxes_a[:, 1]
    a_x2 = boxes_a[:, 2]
    a_y2 = boxes_a[:, 3]
    b_x1 = boxes_b[:, 0]
    b_y1 = boxes_b[:, 1]
    b_x2 = boxes_b[:, 2]
    b_y2 = boxes_b[:, 3]

    # get intersection coordinates
    inter_x1 = np.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = np.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = np.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = np.minimum(a_y2[:, None], b_y2[None, :])

    # get intersection area
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # get union area
    a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    union_area = np.maximum(1e-6, a_area[:, None] + b_area[None, :] - inter_area)

    # iou calc
    iou = inter_area / union_area

    return iou


boxes_a = np.array([[0, 0, 2, 2], [0, 0, 2, 2]])  # 2 boxes
boxes_b = np.array([[1, 1, 3, 3], [3, 3, 5, 5], [0, 0, 2, 2]])  # 3 boxes
print(iou(boxes_a, boxes_b))
