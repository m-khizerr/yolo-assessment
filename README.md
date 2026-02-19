# YOLO Assessment Project

A complete implementation for running YOLO inference using both PyTorch (Ultralytics) and ONNX Runtime, with comparison and validation capabilities.

---

## Steps to Run the Project

### Prerequisites

1. **Clone the repo and go to the project folder:**
   ```bash
   git clone https://github.com/m-khizerr/yolo-assessment.git
   cd yolo-assessment
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Add your model and a sample image** (not in the repo):
   - Place `yolo11n.pt` in the `assets/` folder.
   - Place a sample image (e.g. `image.png`) in the `assets/` folder.

---

### Option A: Run the code (CLI)

Run the full pipeline from the terminal. From the project root (with `venv` activated):

```bash
# 1. PyTorch inference
python src/pytorch_infer.py

# 2. Export model to ONNX (only needed once, or if outputs/ was deleted)
python src/export_onnx.py

# 3. ONNX inference
python src/onnx_infer.py

# 4. Compare PyTorch vs ONNX results
python src/compare_iou.py
```

Results are written to the `outputs/` folder (images and JSON).

---

### Option B: Run the Web UI

Launch the Streamlit app, then upload an image in the browser:

```bash
# Optional: avoid permission issues by using a local Streamlit config dir
mkdir -p .streamlit_home
export STREAMLIT_HOME="$(pwd)/.streamlit_home"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the app
streamlit run src/app.py
```

Open **http://localhost:8501** (or the port shown in the terminal). Upload an image and click **Run Detection** to run PyTorch + ONNX inference and see the comparison. The UI will export the ONNX model automatically if it does not exist.

---

## Project Structure

```
yolo-assessment/
├── assets/
│   ├── yolo11n.pt      # PyTorch YOLO model (place here)
│   └── image.png       # Sample image (place here)
├── outputs/            # Generated outputs (created automatically)
├── src/
│   ├── app.py              # Streamlit web UI
│   ├── pytorch_infer.py    # PyTorch inference script
│   ├── export_onnx.py      # ONNX export script
│   ├── onnx_infer.py       # ONNX inference script
│   ├── compare_iou.py      # Comparison script
│   └── utils/
│       ├── __init__.py
│       ├── io.py           # I/O utilities
│       ├── viz.py           # Visualization utilities
│       ├── preprocess.py    # Preprocessing utilities
│       └── nms.py           # Non-maximum suppression
├── requirements.txt
└── README.md
```

## Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Place Model and Image Files

Place the following files in the `assets/` directory:

- `yolo11n.pt` - PyTorch YOLO model file
- `image.png` - Sample image for inference

```bash
# Ensure assets directory exists
mkdir -p assets

# Copy your files to assets/
# cp /path/to/yolo11n.pt assets/
# cp /path/to/image.png assets/
```

## Usage

### Web UI (Recommended)

Launch the interactive web interface to upload images and run the full pipeline:

```bash
streamlit run src/app.py
```

Then open your browser at `http://localhost:8501`. You can:

1. **Upload** any image (PNG, JPG, JPEG, BMP, WEBP)
2. Click **Run Detection** to run PyTorch + ONNX inference
3. View **results**: annotated images, detection counts, IoU comparison
4. Expand **JSON** sections for raw detection data

The web app will automatically export the ONNX model on first run if it doesn't exist.

---

### CLI Usage

### Step 1: Run PyTorch Inference

Run inference using the PyTorch model:

```bash
python src/pytorch_infer.py
```

**Outputs:**
- `outputs/pytorch_pred.png` - Annotated image with bounding boxes
- `outputs/pytorch_preds.json` - Detection results in JSON format

**Custom options:**
```bash
python src/pytorch_infer.py \
    --model assets/yolo11n.pt \
    --image assets/image.png \
    --outdir outputs \
    --imgsz 640 \
    --conf 0.25 \
    --iou 0.45 \
    --device cpu
```

### Step 2: Export Model to ONNX

Export the PyTorch model to ONNX format:

```bash
python src/export_onnx.py
```

**Outputs:**
- `outputs/yolo11n.onnx` - ONNX model file
- `outputs/onnx_meta.json` - Model metadata (input/output shapes, etc.)

**Custom options:**
```bash
python src/export_onnx.py \
    --model assets/yolo11n.pt \
    --output outputs/yolo11n.onnx \
    --imgsz 640 \
    --opset 13 \
    --dynamic  # Use dynamic axes (optional)
```

### Step 3: Run ONNX Inference

Run inference using the ONNX model (no PyTorch/Ultralytics dependencies):

```bash
python src/onnx_infer.py
```

**Outputs:**
- `outputs/onnx_pred.png` - Annotated image with bounding boxes
- `outputs/onnx_preds.json` - Detection results with runtime breakdown

**Custom options:**
```bash
python src/onnx_infer.py \
    --model outputs/yolo11n.onnx \
    --image assets/image.png \
    --outdir outputs \
    --imgsz 640 \
    --conf 0.25 \
    --iou 0.45
```

### Step 4: Compare Predictions (Recommended)

Compare PyTorch and ONNX predictions:

```bash
python src/compare_iou.py
```

**Outputs:**
- `outputs/compare_report.json` - Detailed comparison report
- `outputs/compare_overlay.png` - Side-by-side visualization (PyTorch=green, ONNX=blue)

**Custom options:**
```bash
python src/compare_iou.py \
    --pytorch outputs/pytorch_preds.json \
    --onnx outputs/onnx_preds.json \
    --out outputs/compare_report.json \
    --iou-threshold 0.5 \
    --image assets/image.png \
    --overlay outputs/compare_overlay.png
```

## Expected Outputs

After running all steps, the `outputs/` directory should contain:

```
outputs/
├── pytorch_pred.png          # PyTorch annotated image
├── pytorch_preds.json        # PyTorch detection results
├── yolo11n.onnx              # Exported ONNX model
├── onnx_meta.json            # ONNX model metadata
├── onnx_pred.png             # ONNX annotated image
├── onnx_preds.json           # ONNX detection results
├── compare_report.json       # Comparison report
└── compare_overlay.png       # Comparison visualization
```

## JSON Output Format

### Detection Results (`*_preds.json`)

```json
{
  "metadata": {
    "model_path": "assets/yolo11n.pt",
    "image_path": "assets/image.png",
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.45,
    "device": "cpu",
    "image_width": 1920,
    "image_height": 1080,
    "runtime_ms": 123.45,
    "num_detections": 5
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.85,
      "bbox_xyxy": [100, 200, 300, 500],
      "bbox_xywh": [200, 350, 200, 300]
    }
  ]
}
```

### Comparison Report (`compare_report.json`)

```json
{
  "summary": {
    "pytorch_detections": 5,
    "onnx_detections": 5,
    "matched_detections": 5,
    "unmatched_pytorch": 0,
    "unmatched_onnx": 0,
    "match_rate_pytorch": 1.0,
    "match_rate_onnx": 1.0,
    "mean_iou": 0.92,
    "median_iou": 0.94,
    "min_iou": 0.85,
    "max_iou": 0.98
  },
  "matched_pairs": [...],
  "unmatched_pytorch": [],
  "unmatched_onnx": []
}
```

## Technical Details

### Preprocessing

Both PyTorch and ONNX inference use consistent preprocessing:

1. **Letterbox Resize**: Resize image to `(imgsz, imgsz)` while maintaining aspect ratio
2. **Padding**: Add gray padding (114, 114, 114) to fill remaining space
3. **Color Conversion**: BGR → RGB
4. **Normalization**: Convert to float32 and normalize to [0, 1]
5. **Layout**: HWC → CHW and add batch dimension → `(1, 3, 640, 640)`

### Postprocessing

1. **Decode Outputs**: Extract boxes, scores, and class IDs from model output
2. **Confidence Filtering**: Remove detections below confidence threshold (default: 0.25)
3. **NMS**: Apply Non-Maximum Suppression with IoU threshold (default: 0.45)
4. **Scale Boxes**: Convert boxes from letterboxed coordinates back to original image coordinates

### ONNX Output Decoding

The ONNX inference script handles multiple output formats robustly:

- **Format Detection**: Automatically detects whether output includes objectness score
  - `(1, 84, 8400)` or `(1, 8400, 84)`: 4 box coords + 80 class scores
  - `(1, 85, 8400)` or `(1, 8400, 85)`: 4 box coords + 1 objectness + 80 class scores

- **Box Format Detection**: Automatically detects box coordinate format
  - **xywh** (center-based): Detected when width/height are positive and centers are within image bounds
  - **xyxy** (corner-based): Used as fallback

- **Confidence Calculation**:
  - With objectness: `confidence = objectness × max(class_probabilities)`
  - Without objectness: `confidence = max(class_probabilities)`

### Default Settings

All scripts use consistent default settings:

- `imgsz`: 640 (image size for inference)
- `conf`: 0.25 (confidence threshold)
- `iou`: 0.45 (IoU threshold for NMS)

These can be overridden via command-line arguments.

## How I Used Cursor

This project was implemented entirely using Cursor AI assistant. Here's how Cursor was leveraged:

- **Code Generation**: Generated all Python scripts, utility modules, and configuration files from scratch
- **Project Structure**: Created the complete directory structure following best practices
- **Error Handling**: Implemented robust error handling and file validation throughout
- **Documentation**: Generated comprehensive README with setup instructions and usage examples
- **Type Hints**: Added type hints and docstrings for better code maintainability
- **Code Review**: Ensured consistency across preprocessing, postprocessing, and visualization utilities
- **Testing Strategy**: Designed scripts to be CLI-friendly with sensible defaults for easy testing

## Troubleshooting

### Model Not Found

Ensure `yolo11n.pt` is placed in `assets/` directory:
```bash
ls assets/yolo11n.pt
```

### Image Not Found

Ensure `image.png` is placed in `assets/` directory:
```bash
ls assets/image.png
```

### ONNX Export Fails

- Ensure PyTorch model is valid
- Check that `onnx` and `onnxsim` packages are installed
- Try reducing `opset` version if compatibility issues occur

### ONNX Inference Mismatches

- Verify preprocessing matches between PyTorch and ONNX paths
- Check that confidence and IoU thresholds are identical
- Review `compare_report.json` for detailed mismatch analysis

### No Detections Found

- Try lowering confidence threshold: `--conf 0.1`
- Verify image contains detectable objects
- Check that model is appropriate for the image content

## Dependencies

- `ultralytics>=8.0.0` - YOLO model loading and PyTorch inference
- `torch>=2.0.0` - PyTorch backend
- `torchvision>=0.15.0` - TorchVision utilities
- `onnx>=1.14.0` - ONNX model format support
- `onnxruntime>=1.15.0` - ONNX Runtime inference engine
- `opencv-python>=4.8.0` - Image processing and visualization
- `numpy>=1.24.0` - Numerical operations
- `tqdm>=4.65.0` - Progress bars (optional)

## License

This project is provided as-is for assessment purposes.
