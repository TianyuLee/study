import numpy as np
from iou import batch_iou

def nms(bboxes: np.ndarray, scores: np.ndarray, threshold):
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        keep.append(order[0])
        bboxes1 = bboxes[order[0]][None, :]
        bboxes2 = bboxes[order[1:]]
        ious = batch_iou(bboxes1, bboxes2)
        inds = np.where(ious < threshold)[0]
        order = order[1 + inds]
    return keep

# 示例数据
boxes = np.array([[100, 100, 210, 210],
                  [250, 250, 420, 420],
                  [100, 100, 210, 210],
                  [200, 200, 300, 300]])

scores = np.array([0.9, 0.8, 0.7, 0.6])

threshold = 0.5

# 调用nms函数
result = nms(boxes, scores, threshold)
print("保留的框的索引:", result)