"""
app.py - Streamlit UI for the dual-stage dental caries research demo.

Run with:
    streamlit run app.py

This file contains ONLY UI logic. All inference is delegated to pipeline.py.
All drawing is delegated to viz.py.
"""

from __future__ import annotations

import io
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pipeline as pl
import viz as vz

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Dental Caries Research Demo",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def find_model_path() -> Path:
    candidates = [
        ROOT / "dental_caries_dualstage_retrained.pkl",
        ROOT / "dental_caries_dualstage_retrrained.pkl",
        ROOT / "dental_caries_dualstage (2).pkl",
        ROOT / "dental_caries_dualstage.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    pkls = list(ROOT.glob("*.pkl"))
    if pkls:
        return pkls[0]
    return ROOT / "dental_caries_dualstage_retrained.pkl"


MODEL_PATH = find_model_path()
SAMPLE_IMAGE_PATH = ROOT / "sample_panoramic.jpg"
MIN_IMAGE_PIXELS = 200 * 200          # reject images smaller than this
MAX_UPLOAD_MB = 50
DISCLAIMER = (
    "⚠️ **Research prototype. Not a medical device. "
    "Not validated for clinical use. "
    "Results must not be used for treatment decisions.**"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_rgb(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to uint8 RGB numpy array."""
    return np.asarray(pil_img.convert("RGB"), dtype=np.uint8)


def _rgb_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode a uint8 RGB numpy array to PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _validate_image(arr: np.ndarray, name: str) -> str | None:
    """Return an error string if the image is unusable, else None."""
    if arr.ndim != 3 or arr.shape[2] != 3:
        return f"'{name}': expected an RGB image."
    h, w = arr.shape[:2]
    if h * w < MIN_IMAGE_PIXELS:
        return (
            f"'{name}' is too small ({w}x{h} px). "
            "Please upload a full-resolution panoramic radiograph "
            "(expected ~2800x1400)."
        )
    return None


def _make_tooth_df(result: pl.InferenceResult) -> pd.DataFrame:
    """Build the per-tooth dataframe from an InferenceResult."""
    rows = []
    for t in result.teeth:
        rows.append(
            {
                "Tooth ID": t.tooth_id,
                "x1": t.box[0],
                "y1": t.box[1],
                "x2": t.box[2],
                "y2": t.box[3],
                "Det Score": round(t.det_score, 4),
                "Lesions": t.n_lesions,
                "Lesion Area (px)": t.total_lesion_area,
                "Max Prob": round(t.max_caries_prob, 4),
                "Mean Prob (in lesion)": round(t.mean_caries_prob, 4),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Max Prob", ascending=False).reset_index(drop=True)
    return df


def _make_json_export(
    filename: str, result: pl.InferenceResult, df: pd.DataFrame
) -> str:
    """Serialise inference results + params to a JSON string."""
    payload = {
        "schema_version": "1.0",
        "source_file": filename,
        "parameters": result.params,
        "versions": {
            "streamlit": st.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "timings": {"inference_ms": round(result.inference_ms, 2)},
        "summary": {
            "teeth_detected": len(result.teeth),
            "lesions_found": sum(t.n_lesions for t in result.teeth),
            "total_lesion_area_px": sum(t.total_lesion_area for t in result.teeth),
            "max_caries_prob": (
                round(max(t.max_caries_prob for t in result.teeth), 4)
                if result.teeth else 0.0
            ),
        },
        "per_tooth": df.to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> dict:
    """Render sidebar controls and return a dict of parameter values."""
    st.sidebar.title("🦷 Pipeline Controls")

    st.sidebar.header("Stage 1 – Tooth Detection")
    det_score_thr = st.sidebar.slider(
        "Detection score threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="Minimum confidence for a detected tooth to be kept.",
        key="det_score_thr",
    )
    nms_iou_thr = st.sidebar.slider(
        "NMS IoU threshold",
        min_value=0.0, max_value=1.0, value=0.30, step=0.01,
        help="Boxes with IoU above this are suppressed. Lower = stricter.",
        key="nms_iou_thr",
    )

    st.sidebar.header("Stage 2 – Caries Segmentation")
    caries_thr = st.sidebar.slider(
        "Caries probability threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="Pixels above this are counted as caries.",
        key="caries_thr",
    )
    min_lesion_area = st.sidebar.slider(
        "Minimum lesion area (px)",
        min_value=1, max_value=2000, value=50, step=1,
        help="Connected components smaller than this are discarded.",
        key="min_lesion_area",
    )

    st.sidebar.header("Pre-processing")
    use_clahe = st.sidebar.toggle(
        "CLAHE on tooth crops", value=True, key="use_clahe",
        help="Apply Contrast Limited Adaptive Histogram Equalisation before Stage 2.",
    )
    use_tta = st.sidebar.toggle(
        "Test-time augmentation (TTA)", value=True, key="use_tta",
        help="Horizontal-flip TTA averages two U-Net passes per crop.",
    )

    st.sidebar.header("Display")
    opacity = st.sidebar.slider(
        "Overlay opacity",
        min_value=0.0, max_value=1.0, value=0.45, step=0.01,
        key="opacity",
    )

    return dict(
        det_score_thr=det_score_thr,
        nms_iou_thr=nms_iou_thr,
        caries_thr=caries_thr,
        min_lesion_area=min_lesion_area,
        use_clahe=use_clahe,
        use_tta=use_tta,
        overlay_opacity=opacity,
    )


# ---------------------------------------------------------------------------
# Model loading + warm-up (runs once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_predictor():
    """Load and warm up the model. Returns (predictor, warmup_ms)."""
    import time
    if not MODEL_PATH.exists():
        return None, 0.0
    predictor = pl.load_model(str(MODEL_PATH), device="auto")
    t0 = time.perf_counter()
    pl.warmup(predictor)
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    return predictor, warmup_ms


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def _render_result(filename: str, result: pl.InferenceResult, params: dict) -> None:
    """Render all five output tabs for one inference result."""
    df = _make_tooth_df(result)
    n_teeth = len(result.teeth)
    n_lesions = sum(t.n_lesions for t in result.teeth)
    total_area = sum(t.total_lesion_area for t in result.teeth)
    max_prob = max((t.max_caries_prob for t in result.teeth), default=0.0)

    # ---- Warnings --------------------------------------------------------
    if n_teeth > 32:
        st.warning(
            f"⚠️ {n_teeth} teeth detected — possible duplicate detections. "
            "Try raising the **NMS IoU threshold**.",
            icon="⚠️",
        )
    if n_lesions > 30:
        st.warning(
            f"⚠️ {n_lesions} lesions found — possibly noisy result. "
            "Try **disabling CLAHE** or raising the **minimum lesion area**.",
            icon="⚠️",
        )

    # ---- Tabs ------------------------------------------------------------
    tab_overlay, tab_heatmap, tab_table, tab_zoom, tab_summary = st.tabs(
        ["🖼 Overlay", "🌡 Heatmap", "📋 Per-tooth table", "🔍 Zoom view", "📊 Summary"]
    )

    # ---- Tab 1: Overlay --------------------------------------------------
    with tab_overlay:
        col_tog1, col_tog2 = st.columns(2)
        show_boxes = col_tog1.checkbox("Show tooth boxes", value=True, key=f"sb_{filename}")
        show_caries = col_tog2.checkbox("Show caries", value=True, key=f"sc_{filename}")
        overlay_img = vz.draw_overlay(
            result.image_rgb,
            result.boxes,
            result.scores,
            result.prob_full,
            caries_thr=params["caries_thr"],
            min_lesion_area=params["min_lesion_area"],
            opacity=params["overlay_opacity"],
            show_boxes=show_boxes,
            show_caries=show_caries,
        )
        st.image(overlay_img, caption=f"Overlay — {filename}", use_container_width=True)
        png_bytes = _rgb_to_png_bytes(overlay_img)
        st.download_button(
            "⬇ Download overlay (PNG)", data=png_bytes,
            file_name=f"{Path(filename).stem}_overlay.png", mime="image/png",
            key=f"dl_overlay_{filename}",
        )

    # ---- Tab 2: Heatmap --------------------------------------------------
    with tab_heatmap:
        heatmap_img = vz.draw_heatmap(
            result.image_rgb, result.prob_full, alpha=params["overlay_opacity"]
        )
        st.image(heatmap_img, caption=f"Caries probability heatmap — {filename}",
                 use_container_width=True)
        # Colorbar
        cbar = vz.probability_colorbar(width=400, height=24)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(cbar, caption="Low ← probability → High", use_container_width=True)
        st.caption("Blue = low probability · Red = high probability")

    # ---- Tab 3: Per-tooth table ------------------------------------------
    with tab_table:
        if df.empty:
            st.info("No teeth detected above the current threshold.")
        else:
            st.dataframe(
                df.style.format(
                    {
                        "Det Score": "{:.4f}",
                        "Max Prob": "{:.4f}",
                        "Mean Prob (in lesion)": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            csv_bytes = df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download table (CSV)", data=csv_bytes,
                file_name=f"{Path(filename).stem}_teeth.csv", mime="text/csv",
                key=f"dl_csv_{filename}",
            )

        json_str = _make_json_export(filename, result, df)
        st.download_button(
            "⬇ Download full results (JSON)", data=json_str.encode(),
            file_name=f"{Path(filename).stem}_results.json", mime="application/json",
            key=f"dl_json_{filename}",
        )

    # ---- Tab 4: Zoom view ------------------------------------------------
    with tab_zoom:
        if df.empty:
            st.info("No teeth detected.")
        else:
            tooth_options = df["Tooth ID"].tolist()
            selected_id = st.selectbox(
                "Select tooth ID", tooth_options,
                key=f"zoom_select_{filename}",
                help="Sorted by max caries probability (highest first).",
            )
            tooth = next(
                (t for t in result.teeth if t.tooth_id == selected_id), None
            )
            if tooth is not None:
                raw_rgb, clahe_rgb, mask_rgb = vz.make_zoom_panel(
                    tooth.crop_raw, tooth.crop_clahe,
                    tooth.prob_map, params["caries_thr"]
                )
                z1, z2, z3 = st.columns(3)
                z1.image(raw_rgb, caption="Raw crop", use_container_width=True)
                z2.image(clahe_rgb, caption="CLAHE crop", use_container_width=True)
                z3.image(mask_rgb, caption="Predicted mask", use_container_width=True)
                st.caption(
                    f"Tooth {selected_id} | "
                    f"Det score: {tooth.det_score:.4f} | "
                    f"Lesions: {tooth.n_lesions} | "
                    f"Max prob: **{tooth.max_caries_prob:.4f}** | "
                    f"Mean prob (in lesion): {tooth.mean_caries_prob:.4f}"
                )

    # ---- Tab 5: Summary --------------------------------------------------
    with tab_summary:
        if n_lesions == 0:
            st.info("No lesions above the current threshold.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Teeth detected", n_teeth)
        m2.metric("Lesions found", n_lesions)
        m3.metric("Total lesion area (px)", f"{total_area:,}")
        m4.metric("Inference time", f"{result.inference_ms:.0f} ms")

        st.caption(
            f"Max raw caries probability across all teeth: **{max_prob:.4f}**"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Permanent disclaimer banner
    st.error(DISCLAIMER, icon="⚕️")

    st.title("🦷 Dental Caries Research Demo")
    st.caption(
        "Dual-Stage Pipeline (Version 2.0 Retrained) · Stage 1: Faster R-CNN (MobileNetV3-Large FPN) "
        "· Stage 2: U-Net (EfficientNet-B0 + CLAHE + TTA + Weighted Own-Box Merging)"
    )

    # Load model
    with st.spinner("Initialising model …"):
        predictor, warmup_ms = _get_predictor()

    if predictor is None:
        st.error(
            f"Model file not found at `{MODEL_PATH}`. "
            "Please ensure `dental_caries_dualstage_retrained.pkl` is present in the app directory.",
            icon="🚫",
        )
        return

    device_str = str(next(predictor.unet.parameters()).device)
    model_version_tag = "v2.0 Retrained" if "retrained" in MODEL_PATH.name.lower() or "retrrained" in MODEL_PATH.name.lower() else "v1.0"
    st.success(
        f"Model **{MODEL_PATH.name}** ({model_version_tag}) ready on **{device_str}** · "
        f"Warm-up completed in {warmup_ms:.0f} ms.",
        icon="✅",
    )

    # Sidebar
    params = _render_sidebar()

    # Session state for results history
    if "results" not in st.session_state:
        st.session_state["results"] = {}   # filename -> InferenceResult

    # ---- Image input section ---------------------------------------------
    st.divider()
    st.subheader("Input")

    col_up, col_sample = st.columns([4, 1])
    with col_up:
        uploaded_files = st.file_uploader(
            "Upload panoramic radiograph(s) — PNG or JPG",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="uploader",
            help=f"Max {MAX_UPLOAD_MB} MB per file. Expected resolution ~2800x1400.",
        )
    with col_sample:
        st.write("")
        st.write("")
        use_sample = st.button(
            "🖼 Use sample image",
            help="Load the bundled sample panoramic for a quick demo.",
            use_container_width=True,
        )

    images_to_process = {}

    if use_sample:
        if not SAMPLE_IMAGE_PATH.exists():
            st.error("Sample image not found in the app directory.", icon="🚫")
        else:
            pil_img = Image.open(SAMPLE_IMAGE_PATH)
            images_to_process["sample_panoramic.jpg"] = _pil_to_rgb(pil_img)

    if uploaded_files:
        for f in uploaded_files:
            try:
                pil_img = Image.open(f)
            except Exception:
                st.error(
                    f"'{f.name}' could not be opened as an image. "
                    "Please upload a valid PNG or JPG file.",
                    icon="🚫",
                )
                continue
            arr = _pil_to_rgb(pil_img)
            err = _validate_image(arr, f.name)
            if err:
                st.error(err, icon="🚫")
                continue
            images_to_process[f.name] = arr

    # ---- Run inference for each image ------------------------------------
    for filename, img_rgb in images_to_process.items():
        # Build a cache key that includes all params
        cache_key = (
            filename,
            params["det_score_thr"],
            params["nms_iou_thr"],
            params["caries_thr"],
            params["min_lesion_area"],
            params["use_clahe"],
            params["use_tta"],
            img_rgb.shape,
            img_rgb.sum(),   # lightweight fingerprint
        )

        stored = st.session_state["results"].get(filename)
        if stored is not None and stored.get("cache_key") == cache_key:
            result = stored["result"]
        else:
            with st.spinner(f"Running inference on {filename} …"):
                try:
                    result = pl.run_inference(
                        img_rgb,
                        predictor,
                        det_score_thr=params["det_score_thr"],
                        nms_iou_thr=params["nms_iou_thr"],
                        caries_thr=params["caries_thr"],
                        min_lesion_area=params["min_lesion_area"],
                        use_clahe=params["use_clahe"],
                        use_tta=params["use_tta"],
                        overlay_opacity=params["overlay_opacity"],
                    )
                except Exception as exc:
                    st.error(
                        f"Inference failed for '{filename}'. "
                        f"Details: {type(exc).__name__}: {exc}",
                        icon="🚫",
                    )
                    continue
            st.session_state["results"][filename] = {
                "cache_key": cache_key,
                "result": result,
            }

        st.divider()
        st.subheader(f"Results — {filename}")
        st.caption(
            f"{result.image_rgb.shape[1]}×{result.image_rgb.shape[0]} px · "
            f"Inference: {result.inference_ms:.0f} ms"
        )
        _render_result(filename, result, params)

    if not images_to_process and not st.session_state["results"]:
        st.info(
            "Upload a panoramic radiograph above, or click **Use sample image** to begin.",
            icon="📂",
        )


if __name__ == "__main__":
    main()
