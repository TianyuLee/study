import numpy as np

def batch_iou(bboxes1, bboxes2):
    """
    input:
        bboxes1: [N, 4], xyxy
        bboxes2: [N, 4], xyxy
    output:
        ious: [N]
    """

    x1 = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
    x2 = np.minimum(bboxes1[:, 2], bboxes2[:, 2])
    y1 = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
    y2 = np.minimum(bboxes1[:, 3], bboxes2[:, 3])
    intersect = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    
    ious = intersect / np.maximum(1e-10, area1 + area2 - intersect)
    return ious

if __name__=='__main__':
    bboxes1 = np.array([[100,100,200,200],[200,200,300,300]])
    bboxes2 = np.array([[150,150,250,250],[250,250,350,350]])
    print(batch_iou(bboxes1, bboxes2))
    print(1 / 7)
    
    