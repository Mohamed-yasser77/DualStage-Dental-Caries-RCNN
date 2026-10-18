# DualStage Dental Caries Detection & Segmentation (Faster R-CNN & U-Net)

A Streamlit research demo for a dual-stage dental caries detection and segmentation pipeline on panoramic radiographs.

> **Research prototype. Not a medical device. Not validated for clinical use. Results must not be used for treatment decisions.**

---

## Pipeline overview

| Stage | Model | Input | Output |
|-------|-------|-------|--------|
| 1 | Faster R-CNN (MobileNetV3-Large FPN) | Full panoramic | Tooth bounding boxes + scores |
| 2 | U-Net (EfficientNet-B0 encoder) | Per-tooth crop (256×256, CLAHE) | Caries probability map |

Predicted per-tooth masks are stitched back to full-image coordinates using a max-pooling strategy.

---

## File layout

```
dental_caries_dualstage (2).pkl   ← model weights (not in repo)
app.py                            ← Streamlit UI
pipeline.py                       ← inference logic (load_model, run_inference)
viz.py                            ← drawing utilities
dental_inference.py               ← original DualStagePredictor class
requirements.txt
sample_panoramic.jpg              ← bundled demo image
```

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

### GPU (CUDA 12.1)

Replace the CPU torch lines in requirements.txt with:

```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

Then install the rest normally.

---

## Running

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Expected input

| Parameter | Value |
|-----------|-------|
| Format | PNG or JPG |
| Resolution | ~2800 × 1400 px (full-mouth panoramic radiograph) |
| Colour | Grayscale or RGB (converted internally) |

Images smaller than 200×200 px are rejected with a clear message.

---

## GPU / CPU notes

- The model loads on **GPU** if CUDA is available, otherwise **CPU**.
- CPU inference on a full panoramic (~2800×1400) typically takes **5–20 s** depending on the number of detected teeth.
- GPU inference is typically **< 1 s**.
- A warm-up pass runs at startup; the reported latency **excludes** this warm-up.

---

## Sidebar controls

| Control | Default | Description |
|---------|---------|-------------|
| Detection score threshold | 0.50 | Stage 1 confidence cutoff |
| NMS IoU threshold | 0.30 | Class-agnostic NMS; lower = stricter duplicate removal |
| Caries probability threshold | 0.50 | Stage 2 binarisation cutoff |
| Minimum lesion area (px) | 50 | Drop small connected components |
| CLAHE toggle | On | Contrast-limited adaptive histogram equalisation on crops |
| TTA toggle | On | Horizontal-flip test-time augmentation |
| Overlay opacity | 0.45 | Alpha blend for caries fill |

---

## Security note

`pipeline.py` loads the model via `pickle.load()` through `DualStagePredictor.from_pickle()`.
This is **safe only when the .pkl file comes from your own training run**.

**TODO (before any public deployment):** convert to `torch.save(state_dict)` /
`torch.load(weights_only=True)`, or export to ONNX.

---

## Output tabs

1. **Overlay** – original image with tooth boxes (yellow, scored) and caries contours (red). Toggle boxes and caries independently.
2. **Heatmap** – continuous probability map blended over the image.
3. **Per-tooth table** – sortable dataframe with per-tooth statistics. Download as CSV.
4. **Zoom view** – raw crop, CLAHE crop, and predicted mask side by side for the selected tooth.
5. **Summary** – aggregate metrics (teeth, lesions, total area, inference time). Download full results as JSON.
