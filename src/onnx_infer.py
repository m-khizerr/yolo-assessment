#!/usr/bin/env python3
"""ONNX inference script using ONNXRuntime (no PyTorch/Ultralytics dependencies)."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from utils.io import load_image, save_json
from utils.nms import non_max_suppression
from utils.preprocess import letterbox, scale_boxes, xywh2xyxy
from utils.viz import COCO_CLASSES, save_annotated_image


def preprocess_image(
    img: np.ndarray,
    imgsz: int = 640,
) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """
    Preprocess image for ONNX inference (matches Ultralytics preprocessing).
    
    Args:
        img: Input image in BGR format (H, W, 3)
        imgsz: Target image size
        
    Returns:
        Tuple of:
        - Preprocessed image (1, 3, H, W) as float32 [0, 1]
        - Scale ratio (scale_x, scale_y)
        - Padding (pad_x, pad_y)
    """
    # Letterbox resize
    img_letterboxed, ratio, pad = letterbox(img, (imgsz, imgsz), auto=False, stride=32)
    
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] and convert to float32
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # HWC -> CHW
    img_chw = img_normalized.transpose(2, 0, 1)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return img_batch, ratio, pad


def decode_yolo_output(
    output: np.ndarray,
    conf_threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode YOLO model output to boxes, scores, and class IDs.
    
    Handles multiple output formats:
    - (1, 84, 8400) or (1, 8400, 84): 4 box coords + 80 class scores
    - (1, 85, 8400) or (1, 8400, 85): 4 box coords + 1 objectness + 80 class scores
    
    Args:
        output: Raw model output
        conf_threshold: Confidence threshold
        
    Returns:
        Tuple of (boxes_xyxy, scores, class_ids)
    """
    print(f"Raw output shape: {output.shape}")
    
    # Handle different output layouts
    if output.ndim == 3:
        batch_size, channels, num_boxes = output.shape
        # Reshape to (num_boxes, channels)
        output = output.transpose(0, 2, 1).reshape(num_boxes, channels)
    elif output.ndim == 2:
        num_boxes, channels = output.shape
    else:
        raise ValueError(f"Unexpected output shape: {output.shape}")
    
    print(f"Reshaped output: {num_boxes} boxes, {channels} channels")
    
    # Detect format based on number of channels
    has_objectness = channels == 85  # 4 box + 1 obj + 80 classes
    
    if has_objectness:
        print("Detected format: 4 box coords + 1 objectness + 80 classes")
        boxes_raw = output[:, :4]  # (N, 4) - box coordinates
        objectness = output[:, 4:5]  # (N, 1) - objectness scores
        class_scores = output[:, 5:]  # (N, 80) - class probabilities
        # Confidence = objectness * max(class_prob)
        confidences = objectness * np.max(class_scores, axis=1, keepdims=True)
    else:
        print("Detected format: 4 box coords + 80 classes (no objectness)")
        boxes_raw = output[:, :4]  # (N, 4) - box coordinates
        class_scores = output[:, 4:]  # (N, 80) - class probabilities
        # Confidence = max(class_prob)
        confidences = np.max(class_scores, axis=1, keepdims=True)
    
    # Get class IDs
    class_ids = np.argmax(class_scores, axis=1)
    
    # Filter by confidence threshold
    mask = confidences.squeeze() >= conf_threshold
    boxes_raw = boxes_raw[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes_raw) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Detect box format (xywh vs xyxy)
    # Check if values look like center-based xywh (width/height should be positive)
    # and centers should be within reasonable bounds for 640x640 image
    sample_box = boxes_raw[0]
    w, h = sample_box[2], sample_box[3]
    
    # Heuristic: if w and h are positive and reasonable, likely xywh
    # Also check if x, y are in center range (roughly 0-640 for 640x640)
    is_xywh = (
        w > 0 and h > 0 and
        w < 640 and h < 640 and
        0 <= sample_box[0] <= 640 and
        0 <= sample_box[1] <= 640
    )
    
    if is_xywh:
        print("Detected box format: xywh (center-based)")
        boxes_xyxy = xywh2xyxy(boxes_raw)
    else:
        print("Detected box format: xyxy (corner-based)")
        boxes_xyxy = boxes_raw.copy()
    
    return boxes_xyxy, confidences.squeeze(), class_ids


def run_onnx_inference(
    model_path: str | Path,
    image_path: str | Path,
    output_dir: str | Path = "outputs",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
) -> None:
    """
    Run ONNX inference using ONNXRuntime.
    
    Args:
        model_path: Path to .onnx model file
        image_path: Path to input image
        output_dir: Output directory for results
        imgsz: Image size for inference
        conf: Confidence threshold
        iou: IoU threshold for NMS
    """
    model_path = Path(model_path)
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    
    # Validate inputs
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    print(f"Loading image from {image_path}...")
    img = load_image(image_path)
    img_height, img_width = img.shape[:2]
    
    # Preprocess
    print("Preprocessing image...")
    preprocess_start = time.time()
    img_preprocessed, ratio, pad = preprocess_image(img, imgsz)
    preprocess_ms = (time.time() - preprocess_start) * 1000
    
    # Create ONNX Runtime session
    print(f"Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(str(model_path))
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Input name: {input_name}, shape: {img_preprocessed.shape}")
    print(f"Output name: {output_name}")
    
    # Run inference
    print("Running inference...")
    infer_start = time.time()
    outputs = session.run([output_name], {input_name: img_preprocessed})
    infer_ms = (time.time() - infer_start) * 1000
    
    output = outputs[0]
    
    # Decode outputs
    print("Decoding outputs...")
    postprocess_start = time.time()
    boxes_xyxy, confidences, class_ids = decode_yolo_output(output, conf)
    
    if len(boxes_xyxy) == 0:
        print("No detections after confidence filtering")
        detections = []
    else:
        # Apply NMS
        print(f"Applying NMS (IoU threshold: {iou})...")
        keep_indices = non_max_suppression(boxes_xyxy, confidences, iou_threshold=iou)
        
        boxes_xyxy = boxes_xyxy[keep_indices]
        confidences = confidences[keep_indices]
        class_ids = class_ids[keep_indices]
        
        # Scale boxes back to original image coordinates
        boxes_xyxy = scale_boxes(
            (imgsz, imgsz),
            boxes_xyxy,
            (img_height, img_width),
            ratio_pad=(ratio[0], pad),
        )
        
        # Convert to detections list
        detections = []
        for i in range(len(boxes_xyxy)):
            box_xyxy = boxes_xyxy[i]
            box_xywh = np.array([
                (box_xyxy[0] + box_xyxy[2]) / 2,  # x_center
                (box_xyxy[1] + box_xyxy[3]) / 2,  # y_center
                box_xyxy[2] - box_xyxy[0],  # width
                box_xyxy[3] - box_xyxy[1],  # height
            ])
            
            class_id = int(class_ids[i])
            confidence = float(confidences[i])
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox_xyxy": box_xyxy.tolist(),
                "bbox_xywh": box_xywh.tolist(),
            })
    
    postprocess_ms = (time.time() - postprocess_start) * 1000
    total_ms = preprocess_ms + infer_ms + postprocess_ms
    
    # Count detections per class
    class_counts = {}
    for det in detections:
        class_name = det["class_name"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Prepare output data
    output_data = {
        "metadata": {
            "model_path": str(model_path),
            "image_path": str(image_path),
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "device": "cpu",  # ONNXRuntime handles device internally
            "image_width": img_width,
            "image_height": img_height,
            "runtime_ms": round(total_ms, 2),
            "preprocess_ms": round(preprocess_ms, 2),
            "infer_ms": round(infer_ms, 2),
            "postprocess_ms": round(postprocess_ms, 2),
            "num_detections": len(detections),
        },
        "detections": detections,
    }
    
    # Save JSON
    json_path = output_dir / "onnx_preds.json"
    save_json(output_data, json_path)
    print(f"Saved predictions to {json_path}")
    
    # Save annotated image
    if len(detections) > 0:
        boxes = np.array([det["bbox_xyxy"] for det in detections])
        labels = [det["class_name"] for det in detections]
        confs = np.array([det["confidence"] for det in detections])
        
        annotated_path = output_dir / "onnx_pred.png"
        save_annotated_image(img, annotated_path, boxes, labels, confs)
        print(f"Saved annotated image to {annotated_path}")
    else:
        # Save original image if no detections
        annotated_path = output_dir / "onnx_pred.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), img)
        print(f"No detections - saved original image to {annotated_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("ONNX Inference Summary")
    print("=" * 50)
    print(f"Preprocess time: {preprocess_ms:.2f} ms")
    print(f"Inference time: {infer_ms:.2f} ms")
    print(f"Postprocess time: {postprocess_ms:.2f} ms")
    print(f"Total runtime: {total_ms:.2f} ms")
    print(f"Total detections: {len(detections)}")
    if class_counts:
        print("\nDetections per class:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ONNX YOLO inference")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/yolo11n.onnx",
        help="Path to ONNX model (.onnx file)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="assets/image.png",
        help="Path to input image",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS",
    )
    
    args = parser.parse_args()
    
    run_onnx_inference(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.outdir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
