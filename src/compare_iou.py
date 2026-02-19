#!/usr/bin/env python3
"""Compare PyTorch and ONNX predictions using IoU matching."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from utils.io import load_json, save_json
from utils.viz import draw_boxes, save_annotated_image


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in xyxy format.
    
    Args:
        box1: Box in xyxy format (4,)
        box2: Box in xyxy format (4,)
        
    Returns:
        IoU value
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_detections(
    pytorch_dets: list[dict],
    onnx_dets: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Match detections between PyTorch and ONNX predictions.
    
    Args:
        pytorch_dets: List of PyTorch detections
        onnx_dets: List of ONNX detections
        iou_threshold: Minimum IoU for matching
        
    Returns:
        Tuple of (matched_pairs, unmatched_pytorch, unmatched_onnx)
    """
    matched_pairs = []
    unmatched_pytorch = list(range(len(pytorch_dets)))
    unmatched_onnx = list(range(len(onnx_dets)))
    
    # Group by class_id for efficiency
    pytorch_by_class = {}
    for i, det in enumerate(pytorch_dets):
        class_id = det["class_id"]
        if class_id not in pytorch_by_class:
            pytorch_by_class[class_id] = []
        pytorch_by_class[class_id].append(i)
    
    onnx_by_class = {}
    for i, det in enumerate(onnx_dets):
        class_id = det["class_id"]
        if class_id not in onnx_by_class:
            onnx_by_class[class_id] = []
        onnx_by_class[class_id].append(i)
    
    # Match detections by class and IoU
    for class_id in set(pytorch_by_class.keys()) | set(onnx_by_class.keys()):
        pytorch_indices = pytorch_by_class.get(class_id, [])
        onnx_indices = onnx_by_class.get(class_id, [])
        
        if not pytorch_indices or not onnx_indices:
            continue
        
        # Compute IoU matrix for this class
        iou_matrix = np.zeros((len(pytorch_indices), len(onnx_indices)))
        for i, pt_idx in enumerate(pytorch_indices):
            for j, onnx_idx in enumerate(onnx_indices):
                box1 = np.array(pytorch_dets[pt_idx]["bbox_xyxy"])
                box2 = np.array(onnx_dets[onnx_idx]["bbox_xyxy"])
                iou_matrix[i, j] = compute_iou(box1, box2)
        
        # Greedy matching: match highest IoU pairs first
        while True:
            if iou_matrix.size == 0:
                break
            
            max_iou = np.max(iou_matrix)
            if max_iou < iou_threshold:
                break
            
            pt_idx, onnx_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            pt_global_idx = pytorch_indices[pt_idx]
            onnx_global_idx = onnx_indices[onnx_idx]
            
            matched_pairs.append({
                "pytorch_idx": pt_global_idx,
                "onnx_idx": onnx_global_idx,
                "iou": float(max_iou),
                "class_id": class_id,
                "pytorch_det": pytorch_dets[pt_global_idx],
                "onnx_det": onnx_dets[onnx_global_idx],
            })
            
            # Remove matched indices
            if pt_global_idx in unmatched_pytorch:
                unmatched_pytorch.remove(pt_global_idx)
            if onnx_global_idx in unmatched_onnx:
                unmatched_onnx.remove(onnx_global_idx)
            
            # Remove from IoU matrix
            iou_matrix = np.delete(iou_matrix, pt_idx, axis=0)
            iou_matrix = np.delete(iou_matrix, onnx_idx, axis=1)
            pytorch_indices = [idx for idx in pytorch_indices if idx != pt_global_idx]
            onnx_indices = [idx for idx in onnx_indices if idx != onnx_global_idx]
    
    unmatched_pytorch_dets = [pytorch_dets[i] for i in unmatched_pytorch]
    unmatched_onnx_dets = [onnx_dets[i] for i in unmatched_onnx]
    
    return matched_pairs, unmatched_pytorch_dets, unmatched_onnx_dets


def compare_predictions(
    pytorch_json: str | Path,
    onnx_json: str | Path,
    output_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compare PyTorch and ONNX predictions.
    
    Args:
        pytorch_json: Path to PyTorch predictions JSON
        onnx_json: Path to ONNX predictions JSON
        output_path: Output report JSON path
        iou_threshold: IoU threshold for matching
        
    Returns:
        Comparison report dictionary
    """
    # Load predictions
    pytorch_data = load_json(pytorch_json)
    onnx_data = load_json(onnx_json)
    
    pytorch_dets = pytorch_data["detections"]
    onnx_dets = onnx_data["detections"]
    
    # Match detections
    matched_pairs, unmatched_pytorch, unmatched_onnx = match_detections(
        pytorch_dets, onnx_dets, iou_threshold
    )
    
    # Compute statistics
    num_matched = len(matched_pairs)
    num_pytorch = len(pytorch_dets)
    num_onnx = len(onnx_dets)
    
    matched_ious = [pair["iou"] for pair in matched_pairs]
    mean_iou = np.mean(matched_ious) if matched_ious else 0.0
    median_iou = np.median(matched_ious) if matched_ious else 0.0
    
    # Create report
    report = {
        "summary": {
            "pytorch_detections": num_pytorch,
            "onnx_detections": num_onnx,
            "matched_detections": num_matched,
            "unmatched_pytorch": len(unmatched_pytorch),
            "unmatched_onnx": len(unmatched_onnx),
            "match_rate_pytorch": num_matched / num_pytorch if num_pytorch > 0 else 0.0,
            "match_rate_onnx": num_matched / num_onnx if num_onnx > 0 else 0.0,
            "mean_iou": float(mean_iou),
            "median_iou": float(median_iou),
            "min_iou": float(np.min(matched_ious)) if matched_ious else 0.0,
            "max_iou": float(np.max(matched_ious)) if matched_ious else 0.0,
        },
        "matched_pairs": matched_pairs,
        "unmatched_pytorch": unmatched_pytorch,
        "unmatched_onnx": unmatched_onnx,
    }
    
    # Save report
    save_json(report, output_path)
    
    return report


def create_comparison_image(
    image_path: str | Path,
    pytorch_json: str | Path,
    onnx_json: str | Path,
    output_path: str | Path,
) -> None:
    """
    Create side-by-side comparison image with PyTorch and ONNX boxes.
    
    Args:
        image_path: Path to original image
        pytorch_json: Path to PyTorch predictions JSON
        onnx_json: Path to ONNX predictions JSON
        output_path: Output image path
    """
    import cv2
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return
    
    pytorch_data = load_json(pytorch_json)
    onnx_data = load_json(onnx_json)
    
    pytorch_dets = pytorch_data["detections"]
    onnx_dets = onnx_data["detections"]
    
    # Draw PyTorch boxes (green)
    if pytorch_dets:
        pytorch_boxes = np.array([det["bbox_xyxy"] for det in pytorch_dets])
        pytorch_labels = [det["class_name"] for det in pytorch_dets]
        pytorch_confs = np.array([det["confidence"] for det in pytorch_dets])
        img = draw_boxes(img, pytorch_boxes, pytorch_labels, pytorch_confs, color=(0, 255, 0))
    
    # Draw ONNX boxes (blue) with slight offset for visibility
    if onnx_dets:
        onnx_boxes = np.array([det["bbox_xyxy"] for det in onnx_dets])
        onnx_labels = [det["class_name"] for det in onnx_dets]
        onnx_confs = np.array([det["confidence"] for det in onnx_dets])
        img = draw_boxes(img, onnx_boxes, onnx_labels, onnx_confs, color=(255, 0, 0))
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Saved comparison image to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX predictions")
    parser.add_argument(
        "--pytorch",
        type=str,
        default="outputs/pytorch_preds.json",
        help="Path to PyTorch predictions JSON",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="outputs/onnx_preds.json",
        help="Path to ONNX predictions JSON",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/compare_report.json",
        help="Output report JSON path",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching detections",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="assets/image.png",
        help="Path to original image (for overlay visualization)",
    )
    parser.add_argument(
        "--overlay",
        type=str,
        default="outputs/compare_overlay.png",
        help="Output path for overlay image",
    )
    
    args = parser.parse_args()
    
    # Compare predictions
    print("Comparing predictions...")
    report = compare_predictions(
        pytorch_json=args.pytorch,
        onnx_json=args.onnx,
        output_path=args.out,
        iou_threshold=args.iou_threshold,
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("Comparison Summary")
    print("=" * 50)
    summary = report["summary"]
    print(f"PyTorch detections: {summary['pytorch_detections']}")
    print(f"ONNX detections: {summary['onnx_detections']}")
    print(f"Matched detections: {summary['matched_detections']}")
    print(f"Unmatched PyTorch: {summary['unmatched_pytorch']}")
    print(f"Unmatched ONNX: {summary['unmatched_onnx']}")
    print(f"\nMatch rate (PyTorch): {summary['match_rate_pytorch']:.2%}")
    print(f"Match rate (ONNX): {summary['match_rate_onnx']:.2%}")
    print(f"\nIoU Statistics:")
    print(f"  Mean IoU: {summary['mean_iou']:.4f}")
    print(f"  Median IoU: {summary['median_iou']:.4f}")
    print(f"  Min IoU: {summary['min_iou']:.4f}")
    print(f"  Max IoU: {summary['max_iou']:.4f}")
    print("=" * 50)
    
    # Create overlay image
    if Path(args.image).exists():
        print(f"\nCreating overlay image...")
        create_comparison_image(args.image, args.pytorch, args.onnx, args.overlay)
    else:
        print(f"\nWarning: Image {args.image} not found, skipping overlay creation")
    
    print(f"\nSaved comparison report to {args.out}")


if __name__ == "__main__":
    main()
