"""Visualization utilities for drawing bounding boxes."""

from pathlib import Path
from typing import List

import cv2
import numpy as np

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_color(class_id: int) -> tuple[int, int, int]:
    """
    Get a color for a class ID.
    
    Args:
        class_id: Class ID (0-79)
        
    Returns:
        BGR color tuple
    """
    # Generate consistent colors using a simple hash
    np.random.seed(class_id)
    color = tuple(map(int, np.random.randint(0, 255, 3)))
    np.random.seed()  # Reset seed
    return color


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: List[str] | None = None,
    confidences: np.ndarray | None = None,
    color: tuple[int, int, int] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: Input image (H, W, 3) in BGR format
        boxes: Boxes in xyxy format (N, 4)
        labels: Optional list of label strings (N,)
        confidences: Optional confidence scores (N,)
        color: Optional color tuple (B, G, R). If None, uses class-based colors
        thickness: Box line thickness
        
    Returns:
        Annotated image
    """
    img = image.copy()
    
    if len(boxes) == 0:
        return img
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Get color
        if color is not None:
            box_color = color
        elif labels is not None and i < len(labels):
            # Try to extract class_id from label if it's a class name
            try:
                class_id = COCO_CLASSES.index(labels[i])
                box_color = get_color(class_id)
            except (ValueError, IndexError):
                box_color = (0, 255, 0)  # Default green
        else:
            box_color = (0, 255, 0)  # Default green
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
        
        # Draw label
        if labels is not None and i < len(labels):
            label = labels[i]
            if confidences is not None and i < len(confidences):
                label = f"{label} {confidences[i]:.2f}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw text background
            cv2.rectangle(
                img,
                (x1, y1 - text_height - baseline - 2),
                (x1 + text_width, y1),
                box_color,
                -1
            )
            
            # Draw text
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
    
    return img


def save_annotated_image(
    image: np.ndarray,
    output_path: str | Path,
    boxes: np.ndarray,
    labels: List[str] | None = None,
    confidences: np.ndarray | None = None,
    color: tuple[int, int, int] | None = None,
) -> None:
    """
    Draw boxes on image and save.
    
    Args:
        image: Input image (H, W, 3) in BGR format
        output_path: Output file path
        boxes: Boxes in xyxy format (N, 4)
        labels: Optional list of label strings
        confidences: Optional confidence scores
        color: Optional color tuple
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    annotated = draw_boxes(image, boxes, labels, confidences, color)
    cv2.imwrite(str(output_path), annotated)
