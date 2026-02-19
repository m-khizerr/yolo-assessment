"""Preprocessing utilities for YOLO inference."""

import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """
    Resize image with letterbox padding to maintain aspect ratio.
    
    Matches Ultralytics letterbox implementation.
    
    Args:
        img: Input image (H, W, C) in BGR format
        new_shape: Target (width, height)
        color: Padding color (B, G, R)
        auto: Auto pad to minimum rectangle
        scale_fill: Stretch image to fill new_shape
        scaleup: Allow upscaling
        stride: Stride for padding
        
    Returns:
        Tuple of:
        - Resized image (H, W, C)
        - Scale ratio (scale_x, scale_y)
        - Padding (pad_x, pad_y)
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2).
    
    Args:
        x: Boxes in xywh format (N, 4) or (4,)
        
    Returns:
        Boxes in xyxy format with same shape
    """
    y = np.copy(x)
    if x.ndim == 1:
        y[0] = x[0] - x[2] / 2  # x1
        y[1] = x[1] - x[3] / 2  # y1
        y[2] = x[0] + x[2] / 2  # x2
        y[3] = x[1] + x[3] / 2  # y2
    else:
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    """
    Convert boxes from (x1, y1, x2, y2) to (x_center, y_center, width, height).
    
    Args:
        x: Boxes in xyxy format (N, 4) or (4,)
        
    Returns:
        Boxes in xywh format with same shape
    """
    y = np.copy(x)
    if x.ndim == 1:
        y[0] = (x[0] + x[2]) / 2  # x_center
        y[1] = (x[1] + x[3]) / 2  # y_center
        y[2] = x[2] - x[0]  # width
        y[3] = x[3] - x[1]  # height
    else:
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x_center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y_center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def scale_boxes(
    img1_shape: tuple[int, int],
    boxes: np.ndarray,
    img0_shape: tuple[int, int],
    ratio_pad: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Scale boxes from letterboxed image coordinates back to original image coordinates.
    
    Args:
        img1_shape: Shape of letterboxed image (height, width)
        boxes: Boxes in xyxy format (N, 4)
        img0_shape: Shape of original image (height, width)
        ratio_pad: Optional (scale_ratio, padding) tuple
        
    Returns:
        Scaled boxes in xyxy format (N, 4)
    """
    if ratio_pad is None:  # calculate from img0_shape and img1_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    
    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    
    # Clip boxes to image bounds
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2
    
    return boxes
