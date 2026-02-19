#!/usr/bin/env python3
"""Streamlit web UI for YOLO inference and comparison."""

import os
import sys
from pathlib import Path

# Ensure we can import from sibling modules when run via streamlit
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Project root (parent of src)
PROJECT_ROOT = _SCRIPT_DIR.parent
WEB_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "web"
ASSETS_DIR = PROJECT_ROOT / "assets"
ONNX_PATH = PROJECT_ROOT / "outputs" / "yolo11n.onnx"
MODEL_PATH = ASSETS_DIR / "yolo11n.pt"

# Streamlit tries to write machine/config state to ~/.streamlit by default.
# In some restricted environments (like sandboxed runs) this can fail with
# PermissionError and cause "Connection lost". Force a writable location.
STREAMLIT_HOME = PROJECT_ROOT / ".streamlit_home"
STREAMLIT_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("STREAMLIT_HOME", str(STREAMLIT_HOME))
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import streamlit as st


def ensure_onnx_exists() -> bool:
    """Export ONNX model if it doesn't exist."""
    if ONNX_PATH.exists():
        return True
    model_pt = ASSETS_DIR / "yolo11n.pt"
    if not model_pt.exists():
        return False
    try:
        from export_onnx import export_to_onnx
        with st.spinner("Exporting ONNX model (one-time setup)..."):
            export_to_onnx(
                model_path=model_pt,
                output_path=ONNX_PATH,
                imgsz=640,
                opset=13,
                dynamic=False,
                simplify=True,
            )
        return True
    except Exception as e:
        st.error(f"Failed to export ONNX: {e}")
        return False


def run_pipeline(image_path: Path) -> dict | None:
    """Run full inference pipeline and return results."""
    WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from pytorch_infer import run_pytorch_inference
        from onnx_infer import run_onnx_inference
        from compare_iou import compare_predictions, create_comparison_image
    except ImportError as e:
        st.error(f"Import error: {e}. Ensure all dependencies are installed.")
        return None

    # Run PyTorch inference
    with st.spinner("Running PyTorch inference..."):
        run_pytorch_inference(
            model_path=MODEL_PATH,
            image_path=image_path,
            output_dir=WEB_OUTPUT_DIR,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            device="cpu",
        )

    # Run ONNX inference
    with st.spinner("Running ONNX inference..."):
        run_onnx_inference(
            model_path=ONNX_PATH,
            image_path=image_path,
            output_dir=WEB_OUTPUT_DIR,
            imgsz=640,
            conf=0.25,
            iou=0.45,
        )

    # Compare
    with st.spinner("Comparing predictions..."):
        report = compare_predictions(
            pytorch_json=WEB_OUTPUT_DIR / "pytorch_preds.json",
            onnx_json=WEB_OUTPUT_DIR / "onnx_preds.json",
            output_path=WEB_OUTPUT_DIR / "compare_report.json",
        )
        create_comparison_image(
            image_path=image_path,
            pytorch_json=WEB_OUTPUT_DIR / "pytorch_preds.json",
            onnx_json=WEB_OUTPUT_DIR / "onnx_preds.json",
            output_path=WEB_OUTPUT_DIR / "compare_overlay.png",
        )

    return {
        "pytorch_pred": WEB_OUTPUT_DIR / "pytorch_pred.png",
        "onnx_pred": WEB_OUTPUT_DIR / "onnx_pred.png",
        "compare_overlay": WEB_OUTPUT_DIR / "compare_overlay.png",
        "pytorch_json": WEB_OUTPUT_DIR / "pytorch_preds.json",
        "onnx_json": WEB_OUTPUT_DIR / "onnx_preds.json",
        "report": report,
    }


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="YOLO Inference & Comparison",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 YOLO Inference & Comparison")
    st.markdown(
        "Upload an image to run **PyTorch** and **ONNX** inference, then compare results."
    )

    # Ensure assets exist
    if not MODEL_PATH.exists():
        st.error(
            f"Model not found: `{MODEL_PATH}`. "
            "Please place `yolo11n.pt` in the `assets/` folder."
        )
        st.stop()

    # Ensure ONNX model exists (export if needed)
    if not ensure_onnx_exists():
        st.error(
            "ONNX model not found and could not be exported. "
            "Run `python src/export_onnx.py` manually first."
        )
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        help="Supported formats: PNG, JPG, JPEG, BMP, WEBP",
    )

    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    # Save uploaded file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    # Show uploaded image
    st.subheader("Uploaded Image")
    st.image(uploaded_file, use_container_width=True)

    # Run button
    if st.button("Run Detection", type="primary"):
        results = run_pipeline(tmp_path)
        tmp_path.unlink(missing_ok=True)

        if results is None:
            return

        # Success message
        st.success("Inference complete!")

        # Metrics
        report = results["report"]
        summary = report["summary"]

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("PyTorch Detections", summary["pytorch_detections"])
        with col2:
            st.metric("ONNX Detections", summary["onnx_detections"])
        with col3:
            st.metric("Matched", summary["matched_detections"])
        with col4:
            st.metric("Match Rate", f"{summary['match_rate_pytorch']:.0%}")
        with col5:
            st.metric("Mean IoU", f"{summary['mean_iou']:.4f}")

        # Result images
        st.subheader("Results")
        img_col1, img_col2, img_col3 = st.columns(3)

        with img_col1:
            st.caption("PyTorch Predictions")
            if results["pytorch_pred"].exists():
                st.image(str(results["pytorch_pred"]), use_container_width=True)

        with img_col2:
            st.caption("ONNX Predictions")
            if results["onnx_pred"].exists():
                st.image(str(results["onnx_pred"]), use_container_width=True)

        with img_col3:
            st.caption("Comparison Overlay (PyTorch=Green, ONNX=Blue)")
            if results["compare_overlay"].exists():
                st.image(str(results["compare_overlay"]), use_container_width=True)

        # Expandable JSON
        with st.expander("PyTorch detections (JSON)"):
            from utils.io import load_json
            pt_data = load_json(results["pytorch_json"])
            st.json(pt_data)

        with st.expander("ONNX detections (JSON)"):
            from utils.io import load_json
            onnx_data = load_json(results["onnx_json"])
            st.json(onnx_data)

        with st.expander("Comparison report (JSON)"):
            st.json(report)


if __name__ == "__main__":
    main()
