# Visual overlay and colorbar styling for dental panoramic radiographs
"""
viz.py - Drawing and visualisation utilities for the dental caries Streamlit app.

All functions operate on numpy arrays and return uint8 RGB images
suitable for st.image.
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Colour constants (RGB)
# ---------------------------------------------------------------------------

TOOTH_BOX_COLOR = (255, 210, 0)       # yellow
CARIES_FILL_COLOR = (220, 30, 30)     # red fill
CARIES_CONTOUR_COLOR = (255, 40, 40)  # bright red outline


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def draw_overlay(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    prob_full: np.ndarray,
    caries_thr: float,
    min_lesion_area: int,
    opacity: float = 0.45,
    show_boxes: bool = True,
    show_caries: bool = True,
) -> np.ndarray:
    """
    Draw tooth bounding boxes and caries contours on the image.

    Parameters
    ----------
    image_rgb : uint8 [H,W,3] source image.
    boxes     : float32 [N,4] tooth boxes (x1,y1,x2,y2).
    scores    : float32 [N] detection scores.
    prob_full : float32 [H,W] full-image probability map.
    caries_thr: Threshold for binarising the map.
    min_lesion_area: Minimum component area to draw.
    opacity   : Alpha blend weight for caries fill.
    show_boxes: Draw tooth bounding boxes.
    show_caries: Draw caries fill and contours.

    Returns
    -------
    uint8 [H,W,3] annotated RGB image.
    """
    vis = image_rgb.copy()

    if show_caries and prob_full is not None:
        binary = (prob_full >= caries_thr).astype(np.uint8)
        if min_lesion_area > 1:
            n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            clean = np.zeros_like(binary)
            for lbl in range(1, n_lbl):
                if stats[lbl, cv2.CC_STAT_AREA] >= min_lesion_area:
                    clean[labels == lbl] = 1
            binary = clean

        if binary.max() > 0:
            fill = vis.copy()
            fill[binary > 0] = CARIES_FILL_COLOR
            vis = cv2.addWeighted(fill, opacity, vis, 1.0 - opacity, 0)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # OpenCV uses BGR; reverse our RGB tuples
            cv2.drawContours(
                vis, contours, -1, CARIES_CONTOUR_COLOR[::-1], 2
            )

    if show_boxes and boxes is not None and len(boxes) > 0:
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(
                vis, (x1, y1), (x2, y2), TOOTH_BOX_COLOR[::-1], 2
            )
            label = f"{sc:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
            )
            lx = x1 + 2
            ly = max(y1 - 4, th + 4)
            # Score badge background
            cv2.rectangle(
                vis,
                (lx - 1, ly - th - 2),
                (lx + tw + 1, ly + 2),
                TOOTH_BOX_COLOR[::-1],
                -1,
            )
            cv2.putText(
                vis, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA
            )

    return vis


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def draw_heatmap(
    image_rgb: np.ndarray,
    prob_full: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a JET-colourmap probability heatmap over the original image.

    Parameters
    ----------
    image_rgb : uint8 [H,W,3] source image.
    prob_full : float32 [H,W] probability map in [0,1].
    alpha     : Heatmap blend weight (0=invisible, 1=opaque).

    Returns
    -------
    uint8 [H,W,3] blended RGB image.
    """
    prob_u8 = (np.clip(prob_full, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    bgr_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    blended_bgr = cv2.addWeighted(heatmap_bgr, alpha, bgr_img, 1.0 - alpha, 0)
    return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Zoom / crop panel
# ---------------------------------------------------------------------------

def make_zoom_panel(
    crop_raw: np.ndarray,
    crop_clahe: np.ndarray,
    prob_map: np.ndarray,
    caries_thr: float,
) -> tuple:
    """
    Produce three uint8 RGB images for the zoom view.

    Parameters
    ----------
    crop_raw   : uint8 [H,W] grayscale raw crop.
    crop_clahe : uint8 [H,W] grayscale CLAHE-processed crop.
    prob_map   : float32 [H,W] U-Net probability map.
    caries_thr : Binarisation threshold for the mask panel.

    Returns
    -------
    (raw_rgb, clahe_rgb, mask_rgb) — each uint8 [H,W,3].
    """
    raw_rgb = cv2.cvtColor(crop_raw, cv2.COLOR_GRAY2RGB)
    clahe_rgb = cv2.cvtColor(crop_clahe, cv2.COLOR_GRAY2RGB)

    mask_base = cv2.cvtColor(
        (np.clip(prob_map, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_GRAY2RGB,
    )
    binary = (prob_map >= caries_thr).astype(np.uint8)
    if binary.max() > 0:
        fill = mask_base.copy()
        fill[binary > 0] = CARIES_FILL_COLOR
        mask_base = cv2.addWeighted(fill, 0.5, mask_base, 0.5, 0)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(mask_base, contours, -1, CARIES_CONTOUR_COLOR, 1)

    return raw_rgb, clahe_rgb, mask_base


# ---------------------------------------------------------------------------
# Colorbar legend
# ---------------------------------------------------------------------------

def probability_colorbar(width: int = 300, height: int = 20) -> np.ndarray:
    """
    Return a uint8 [height, width, 3] RGB JET colorbar (0 -> 1).
    """
    bar = np.tile(
        (np.arange(width, dtype=np.float32) / width * 255).astype(np.uint8),
        (height, 1),
    )
    colored = cv2.applyColorMap(bar, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
