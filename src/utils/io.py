"""I/O utilities for loading images and saving JSON results."""

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image using OpenCV.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as BGR numpy array (H, W, 3)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return img


def save_json(data: Dict[str, Any], output_path: str | Path, indent: int = 2) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        output_path: Output file path
        indent: JSON indentation level
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(json_path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Loaded dictionary
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        return json.load(f)
