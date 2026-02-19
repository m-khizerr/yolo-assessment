#!/usr/bin/env python3
"""Export PyTorch YOLO model to ONNX format."""

import argparse
import json
from pathlib import Path

import onnx
from ultralytics import YOLO

from utils.io import save_json


def export_to_onnx(
    model_path: str | Path,
    output_path: str | Path,
    imgsz: int = 640,
    opset: int = 13,
    dynamic: bool = False,
    simplify: bool = True,
) -> None:
    """
    Export YOLO model to ONNX format.
    
    Args:
        model_path: Path to .pt model file
        output_path: Output .onnx file path
        imgsz: Image size for export
        opset: ONNX opset version
        dynamic: Use dynamic axes
        simplify: Apply ONNX simplifier
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))
    
    # Export to ONNX
    print(f"Exporting to ONNX (opset={opset}, imgsz={imgsz}, dynamic={dynamic})...")
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        nms=False,  # Export raw predictions, we'll do NMS ourselves
    )
    
    # Find the exported ONNX file (Ultralytics saves it next to the .pt file)
    exported_path = model_path.with_suffix(".onnx")
    if not exported_path.exists():
        raise FileNotFoundError(f"Exported ONNX file not found at {exported_path}")
    
    # Move to desired output location
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path != output_path:
        import shutil
        shutil.move(str(exported_path), str(output_path))
        print(f"Moved ONNX file to {output_path}")
    else:
        print(f"ONNX file saved to {output_path}")
    
    # Validate ONNX model
    print("Validating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Get model metadata
    import onnxruntime as ort
    session = ort.InferenceSession(str(output_path))
    
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    
    print(f"\nONNX Model Info:")
    print(f"  Input name: {input_meta.name}")
    print(f"  Input shape: {input_meta.shape}")
    print(f"  Input type: {input_meta.type}")
    print(f"  Output name: {output_meta.name}")
    print(f"  Output shape: {output_meta.shape}")
    print(f"  Output type: {output_meta.type}")
    
    # Save metadata
    metadata = {
        "input_name": input_meta.name,
        "input_shape": input_meta.shape,
        "input_type": input_meta.type,
        "output_name": output_meta.name,
        "output_shape": output_meta.shape,
        "output_type": output_meta.type,
        "imgsz": imgsz,
        "opset": opset,
        "dynamic": dynamic,
        "simplify": simplify,
        "export_args": {
            "imgsz": imgsz,
            "opset": opset,
            "dynamic": dynamic,
            "simplify": simplify,
            "nms": False,
        },
    }
    
    meta_path = output_path.parent / "onnx_meta.json"
    save_json(metadata, meta_path)
    print(f"\nSaved metadata to {meta_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="assets/yolo11n.pt",
        help="Path to PyTorch model (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/yolo11n.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for export",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use dynamic axes",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX simplifier",
    )
    
    args = parser.parse_args()
    
    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=not args.no_simplify,
    )


if __name__ == "__main__":
    main()
