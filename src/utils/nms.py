"""Non-Maximum Suppression implementation using pure NumPy."""

import numpy as np


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        boxes: Boxes in xyxy format (N, 4)
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Sort by score (descending)
    order = scores.argsort()[::-1]
    keep = []
    
    while len(order) > 0:
        # Take the box with highest score
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate IoU with remaining boxes
        ious = compute_iou(boxes[i:i+1], boxes[order[1:]])
        
        # Keep boxes with IoU < threshold
        mask = ious < iou_threshold
        order = order[1:][mask]
    
    return np.array(keep, dtype=np.int32)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between boxes.
    
    Args:
        box1: Boxes in xyxy format (M, 4)
        box2: Boxes in xyxy format (N, 4)
        
    Returns:
        IoU matrix (M, N)
    """
    # Expand dimensions for broadcasting
    box1 = box1[:, None, :]  # (M, 1, 4)
    box2 = box2[None, :, :]  # (1, N, 4)
    
    # Calculate intersection
    x1 = np.maximum(box1[:, :, 0], box2[:, :, 0])
    y1 = np.maximum(box1[:, :, 1], box2[:, :, 1])
    x2 = np.minimum(box1[:, :, 2], box2[:, :, 2])
    y2 = np.minimum(box1[:, :, 3], box2[:, :, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate union
    area1 = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1])
    area2 = (box2[:, :, 2] - box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1])
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    iou = intersection / np.maximum(union, 1e-6)
    
    return iou.squeeze() if iou.shape[0] == 1 else iou
