#!/usr/bin/env python3
"""PyTorch inference script using Ultralytics YOLO."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from utils.io import load_image, save_json
from utils.viz import COCO_CLASSES, save_annotated_image


def run_pytorch_inference(
    model_path: str | Path,
    image_path: str | Path,
    output_dir: str | Path = "outputs",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
) -> None:
    """
    Run PyTorch inference using Ultralytics YOLO.
    
    Args:
        model_path: Path to .pt model file
        image_path: Path to input image
        output_dir: Output directory for results
        imgsz: Image size for inference
        conf: Confidence threshold
        iou: IoU threshold for NMS
        device: Device to use ('cpu' or 'cuda')
    """
    model_path = Path(model_path)
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    
    # Validate inputs
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))
    
    # Load image to get dimensions
    img = load_image(image_path)
    img_height, img_width = img.shape[:2]
    
    # Run inference
    print(f"Running inference on {image_path}...")
    start_time = time.time()
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    runtime_ms = (time.time() - start_time) * 1000
    
    # Process results
    if len(results) == 0:
        print("Warning: No detections found")
        detections = []
    else:
        result = results[0]
        detections = []
        
        for box in result.boxes:
            # Get box coordinates (already in original image coordinates)
            xyxy = box.xyxy[0].cpu().numpy()
            xywh = box.xywh[0].cpu().numpy()
            
            # Get class and confidence
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox_xyxy": xyxy.tolist(),
                "bbox_xywh": xywh.tolist(),
            })
    
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
            "device": device,
            "image_width": img_width,
            "image_height": img_height,
            "runtime_ms": round(runtime_ms, 2),
            "num_detections": len(detections),
        },
        "detections": detections,
    }
    
    # Save JSON
    json_path = output_dir / "pytorch_preds.json"
    save_json(output_data, json_path)
    print(f"Saved predictions to {json_path}")
    
    # Save annotated image
    if len(detections) > 0:
        boxes = np.array([det["bbox_xyxy"] for det in detections])
        labels = [det["class_name"] for det in detections]
        confidences = np.array([det["confidence"] for det in detections])
        
        annotated_path = output_dir / "pytorch_pred.png"
        save_annotated_image(img, annotated_path, boxes, labels, confidences)
        print(f"Saved annotated image to {annotated_path}")
    else:
        # Save original image if no detections
        import cv2
        annotated_path = output_dir / "pytorch_pred.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), img)
        print(f"No detections - saved original image to {annotated_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("PyTorch Inference Summary")
    print("=" * 50)
    print(f"Runtime: {runtime_ms:.2f} ms")
    print(f"Total detections: {len(detections)}")
    if class_counts:
        print("\nDetections per class:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run PyTorch YOLO inference")
    parser.add_argument(
        "--model",
        type=str,
        default="assets/yolo11n.pt",
        help="Path to PyTorch model (.pt file)",
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
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    run_pytorch_inference(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.outdir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )


if __name__ == "__main__":
    main()
