"""
pipeline.py - Inference wrapper for the dual-stage dental caries pipeline.

Public API
----------
    load_model(path, device)     -> DualStagePredictor   (cached by Streamlit)
    run_inference(image, params) -> InferenceResult

Security note
-------------
The weights are loaded via pickle.load() through DualStagePredictor.from_pickle().
This is safe only when the .pkl file comes from a trusted source (your own training run).
TODO: Convert to torch.save(state_dict) / torch.load(weights_only=True) or export to ONNX
      before any public / internet-facing deployment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch

from dental_inference import DualStagePredictor, prep


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToothResult:
    """Per-tooth inference results."""
    tooth_id: int
    box: tuple                          # (x1, y1, x2, y2) original coords
    padded_box: tuple                   # box used for crop
    det_score: float
    crop_raw: np.ndarray                # uint8 [H,W] grayscale, no CLAHE
    crop_clahe: np.ndarray              # uint8 [H,W] grayscale, CLAHE applied
    prob_map: np.ndarray                # float32 [256,256] caries probability
    n_lesions: int
    total_lesion_area: int              # pixels in full-image coords
    max_caries_prob: float
    mean_caries_prob: float             # mean inside lesion mask only


@dataclass
class InferenceResult:
    """Full result for one image."""
    image_rgb: np.ndarray               # uint8 [H,W,3] original image
    prob_full: np.ndarray               # float32 [H,W] full-image probability map
    boxes: np.ndarray                   # float32 [N,4]
    scores: np.ndarray                  # float32 [N]
    teeth: list = field(default_factory=list)
    inference_ms: float = 0.0
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model weights ...")
def load_model(pkl_path: str, device: str = "auto") -> DualStagePredictor:
    """
    Load the dual-stage predictor from a pickle bundle.

    Parameters
    ----------
    pkl_path:
        Absolute path to dental_caries_dualstage.pkl.
    device:
        'auto' -> GPU if available, else CPU.  'cuda' or 'cpu' to force.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DualStagePredictor.from_pickle(str(pkl_path), device=device)
    predictor.unet.eval()
    predictor.det.eval()
    return predictor


# ---------------------------------------------------------------------------
# NMS helper (class-agnostic)
# ---------------------------------------------------------------------------

def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Return kept indices after class-agnostic greedy NMS."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, ix2 - ix1)
        h = np.maximum(0.0, iy2 - iy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# Crop utilities (CLAHE-optional)
# ---------------------------------------------------------------------------

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _expand_box(box: np.ndarray, W: int, H: int, pad: float = 0.15):
    """Expand box by `pad` fraction and clip to image bounds."""
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(np.floor(x1 - pad * bw)))
    y1 = max(0, int(np.floor(y1 - pad * bh)))
    x2 = min(W, int(np.ceil(x2 + pad * bw)))
    y2 = min(H, int(np.ceil(y2 + pad * bh)))
    return x1, y1, x2, y2


def _make_crop(gray: np.ndarray, box: np.ndarray, size: int = 256,
               pad: float = 0.15, use_clahe: bool = True):
    """
    Return (raw_crop, clahe_crop, padded_box) or (None, None, None) on error.
    Both crops are uint8 [size, size] grayscale.
    """
    H, W = gray.shape[:2]
    pb = _expand_box(box, W, H, pad)
    x1, y1, x2, y2 = pb
    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return None, None, None
    region = gray[y1:y2, x1:x2]
    interp = cv2.INTER_AREA if min(region.shape[:2]) > size else cv2.INTER_LINEAR
    raw = cv2.resize(region, (size, size), interpolation=interp)
    if use_clahe:
        clahe_reg = _CLAHE.apply(region)
        clahe_crop = cv2.resize(clahe_reg, (size, size), interpolation=interp)
    else:
        clahe_crop = raw.copy()
    return raw, clahe_crop, pb


# ---------------------------------------------------------------------------
# Segmentation with TTA
# ---------------------------------------------------------------------------

@torch.no_grad()
def _segment_batch(predictor: DualStagePredictor, crops_u8: np.ndarray,
                   use_tta: bool = True, bs: int = 32) -> np.ndarray:
    """
    Run U-Net on a batch of uint8 [N,H,W] grayscale crops.
    Returns float32 [N,H,W] probability maps in [0,1].
    """
    out = []
    for i in range(0, len(crops_u8), bs):
        batch = np.ascontiguousarray(crops_u8[i: i + bs])
        xb = (torch.from_numpy(batch).float().div(255.0)
              .unsqueeze(1).to(predictor.device))
        lg = predictor.unet(prep(xb)).float()
        if use_tta:
            flipped = torch.flip(xb, dims=[-1])
            lg_f = torch.flip(predictor.unet(prep(flipped)).float(), dims=[-1])
            lg = (lg + lg_f) / 2.0
        prob = torch.sigmoid(lg).squeeze(1).cpu().numpy()
        out.append(prob)
    return np.concatenate(out) if out else np.zeros((0, 256, 256), np.float32)


# ---------------------------------------------------------------------------
# Lesion post-processing
# ---------------------------------------------------------------------------

def _postprocess_mask(prob_region: np.ndarray, caries_thr: float,
                      min_lesion_area: int) -> np.ndarray:
    """
    Threshold -> morphological opening -> remove small connected components.
    Returns uint8 binary mask [H,W].
    """
    binary = (prob_region >= caries_thr).astype(np.uint8)
    if binary.max() == 0:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if min_lesion_area > 1:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        clean = np.zeros_like(binary)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_lesion_area:
                clean[labels == lbl] = 1
        binary = clean
    return binary


# ---------------------------------------------------------------------------
# Main inference entry point
# ---------------------------------------------------------------------------

def run_inference(
    image_rgb: np.ndarray,
    predictor: DualStagePredictor,
    det_score_thr: float = 0.5,
    nms_iou_thr: float = 0.3,
    caries_thr: float = 0.5,
    min_lesion_area: int = 50,
    crop_pad: float = 0.15,
    use_clahe: bool = True,
    use_tta: bool = True,
    overlay_opacity: float = 0.45,
) -> InferenceResult:
    """
    Run the full dual-stage pipeline on a single RGB image (uint8 [H,W,3]).

    Parameters
    ----------
    image_rgb:
        Input panoramic radiograph as uint8 RGB numpy array.
    predictor:
        Loaded DualStagePredictor instance.
    det_score_thr:
        Tooth detection confidence threshold for Stage 1.
    nms_iou_thr:
        IoU threshold for class-agnostic NMS applied after Stage 1.
    caries_thr:
        Probability threshold for binarising the caries mask.
    min_lesion_area:
        Connected components smaller than this (pixels) are discarded.
    crop_pad:
        Fractional padding added to each detected tooth box before cropping.
    use_clahe:
        Whether to apply CLAHE to tooth crops before Stage 2.
    use_tta:
        Whether to apply horizontal-flip test-time augmentation in Stage 2.
    overlay_opacity:
        Blending weight stored in params (used by viz layer).

    Returns
    -------
    InferenceResult with all per-tooth data and the full-image probability map.
    """
    t0 = time.perf_counter()

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape

    # ---- Stage 1: tooth detection ----------------------------------------
    with torch.no_grad():
        x = (torch.from_numpy(gray).float().div(255.0)
             .unsqueeze(0).repeat(3, 1, 1).to(predictor.device))
        raw_out = predictor.det([x])[0]

    all_boxes = raw_out["boxes"].float().cpu().numpy()
    all_scores = raw_out["scores"].float().cpu().numpy()

    # Score threshold
    keep = all_scores >= det_score_thr
    boxes = all_boxes[keep]
    scores = all_scores[keep]

    # Class-agnostic NMS
    if len(boxes) > 0:
        nms_keep = _nms_boxes(boxes, scores, nms_iou_thr)
        boxes = boxes[nms_keep]
        scores = scores[nms_keep]

    # ---- Stage 2: per-tooth segmentation ----------------------------------
    raw_crops = []
    clahe_crops = []
    padded_boxes = []
    valid_boxes = []
    valid_scores = []

    for box, score in zip(boxes, scores):
        raw_c, clahe_c, pb = _make_crop(
            gray, box, size=predictor.size, pad=crop_pad, use_clahe=use_clahe
        )
        if raw_c is None:
            continue
        raw_crops.append(raw_c)
        clahe_crops.append(clahe_c)
        padded_boxes.append(pb)
        valid_boxes.append(box)
        valid_scores.append(float(score))

    # Segment using the processed (CLAHE or raw) crops
    seg_input = np.stack(clahe_crops) if clahe_crops else np.zeros((0, 256, 256), np.uint8)
    crop_probs = _segment_batch(predictor, seg_input, use_tta=use_tta)

    # ---- Stitch probability map ------------------------------------------
    prob_full = np.zeros((H, W), np.float32)
    for crop_prob, pb in zip(crop_probs, padded_boxes):
        x1, y1, x2, y2 = pb
        resized = cv2.resize(crop_prob, (x2 - x1, y2 - y1),
                             interpolation=cv2.INTER_LINEAR)
        prob_full[y1:y2, x1:x2] = np.maximum(prob_full[y1:y2, x1:x2], resized)

    # ---- Per-tooth stats --------------------------------------------------
    teeth = []
    for tid, (box, score, crop_prob, raw_c, clahe_c, pb) in enumerate(
        zip(valid_boxes, valid_scores, crop_probs, raw_crops, clahe_crops, padded_boxes)
    ):
        x1p, y1p, x2p, y2p = pb
        region_prob = cv2.resize(crop_prob, (x2p - x1p, y2p - y1p),
                                 interpolation=cv2.INTER_LINEAR)
        region_mask = _postprocess_mask(region_prob, caries_thr, min_lesion_area)

        n_labels, _, _, _ = cv2.connectedComponentsWithStats(region_mask)
        n_lesions = max(0, n_labels - 1)
        total_area = int(region_mask.sum())
        max_prob = float(region_prob.max()) if region_prob.size > 0 else 0.0
        lesion_pixels = region_prob[region_mask > 0]
        mean_prob = float(lesion_pixels.mean()) if lesion_pixels.size > 0 else 0.0

        teeth.append(ToothResult(
            tooth_id=tid,
            box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
            padded_box=pb,
            det_score=score,
            crop_raw=raw_c,
            crop_clahe=clahe_c,
            prob_map=crop_prob,
            n_lesions=n_lesions,
            total_lesion_area=total_area,
            max_caries_prob=max_prob,
            mean_caries_prob=mean_prob,
        ))

    inference_ms = (time.perf_counter() - t0) * 1000.0

    params = dict(
        det_score_thr=det_score_thr,
        nms_iou_thr=nms_iou_thr,
        caries_thr=caries_thr,
        min_lesion_area=min_lesion_area,
        crop_pad=crop_pad,
        use_clahe=use_clahe,
        use_tta=use_tta,
        overlay_opacity=overlay_opacity,
    )

    return InferenceResult(
        image_rgb=image_rgb,
        prob_full=prob_full,
        boxes=np.asarray(valid_boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.asarray(valid_scores, dtype=np.float32),
        teeth=teeth,
        inference_ms=inference_ms,
        params=params,
    )


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

def warmup(predictor: DualStagePredictor) -> None:
    """
    Run a single forward pass on a tiny dummy image to warm up CUDA/JIT caches.
    Call once after load_model() before handling real requests.
    """
    dummy_rgb = np.zeros((128, 256, 3), dtype=np.uint8)
    run_inference(
        dummy_rgb,
        predictor,
        det_score_thr=0.99,
        nms_iou_thr=0.3,
        caries_thr=0.5,
        min_lesion_area=1,
        use_clahe=False,
        use_tta=False,
    )
