"""
test.py - Dental Panoramic Caries Detection & Tooth Localization CLI

Dual-Stage Deep Learning Pipeline:
  Stage 1: Faster R-CNN (MobileNetV3-Large) for Tooth Localization (Yellow Bounding Boxes)
  Stage 2: U-Net (EfficientNet-B0 + CLAHE + TTA) for Caries Segmentation (Red Masks & Contours)

Usage:
  # Test a single image:
  python test.py --image "path/to/panoramic_xray.png"

  # Test all images in a directory:
  python test.py --input-dir "path/to/images_folder"

  # Custom output directory:
  python test.py --image "xray.png" --out-dir "my_results"

  # Quick test with default samples:
  python test.py
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

# Ensure local module import works regardless of where the script is called from
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from dental_inference import DualStagePredictor


def find_model_path():
    """Look for the dual-stage pickle bundle in the project directory."""
    candidates = [
        BASE_DIR / "dental_caries_dualstage (2).pkl",
        BASE_DIR / "dental_caries_dualstage.pkl",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Search for any .pkl in current directory
    pkls = list(BASE_DIR.glob("*.pkl"))
    if pkls:
        return str(pkls[0])
    raise FileNotFoundError(
        "Could not find model weights (.pkl file). Please place "
        "'dental_caries_dualstage (2).pkl' in the same folder as test.py."
    )


def process_image(predictor, image_path, out_dir, det_thr=None, seg_thr=None):
    """Run dual-stage inference on an image and save detection overlay & heatmap."""
    image_path = Path(image_path)
    if not image_path.is_file():
        print(f"[Error] Image not found: {image_path}")
        return

    raw_img = cv2.imread(str(image_path))
    if raw_img is None:
        print(f"[Error] Could not read image (corrupt or unsupported format): {image_path}")
        return

    H, W = raw_img.shape[:2]
    stem = image_path.stem
    print(f"\n=======================================================")
    print(f"Processing: {image_path.name} ({W}x{H} px)")

    # 1. Run inference
    result = predictor.predict(str(image_path))
    boxes = result["boxes"]
    scores = result["scores"]
    mask = result["mask"]
    prob = result["prob"]

    # Filter by custom threshold if specified
    if seg_thr is not None:
        mask = (prob > seg_thr).astype(np.uint8)

    # 2. Extract Caries Contours
    caries_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(caries_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_prob = float(np.nanmax(prob)) if prob.size > 0 else 0.0
    positive_pixels = int(mask.sum())

    print(f" -> Teeth detected: {len(boxes)}")
    print(f" -> Caries lesions detected: {len(contours)}")
    print(f" -> Peak caries confidence: {max_prob * 100:.2f}%")
    print(f" -> Affected pixel area: {positive_pixels} px")

    # 3. Create Visual Overlay (Yellow BBoxes + Red Mask/Contours)
    vis = raw_img.copy()

    # Draw Yellow Tooth Bounding Boxes
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 240, 255), 2)
        cv2.putText(
            vis, f"{sc:.2f}", (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA
        )

    # Draw Red Caries Overlay and Border
    red_layer = vis.copy()
    red_layer[mask > 0] = [0, 0, 255]  # Bright Red (BGR)
    vis = cv2.addWeighted(red_layer, 0.45, vis, 0.55, 0)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

    # Top Banner Header
    banner_h = 75
    banner = np.zeros((banner_h, W, 3), dtype=np.uint8)
    cv2.putText(
        banner, f"Dental R-CNN & U-Net Caries Analysis: {image_path.name}",
        (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        banner,
        f"Teeth: {len(boxes)} | Caries Lesions: {len(contours)} | Max Conf: {max_prob*100:.1f}% | Yellow: Tooth Box | Red: Caries",
        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 1, cv2.LINE_AA
    )

    vis_final = np.vstack([banner, vis])
    det_out_path = out_dir / f"{stem}_detection_overlay.jpg"
    cv2.imwrite(str(det_out_path), vis_final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f" -> Saved Detection Overlay : {det_out_path}")

    # 4. Generate Continuous Probability Heatmap
    p_clamped = np.nan_to_num(prob, 0.0)
    prob_u8 = (np.clip(p_clamped, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    heatmap_blended = cv2.addWeighted(raw_img, 0.65, heatmap, 0.35, 0)

    hm_banner = np.zeros((banner_h, W, 3), dtype=np.uint8)
    cv2.putText(
        hm_banner, f"Caries Probability Heatmap: {image_path.name}",
        (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        hm_banner,
        "Color Gradient: Blue (Healthy) -> Green/Yellow (Suspicious) -> Bright Red (High Confidence Caries)",
        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA
    )

    heatmap_final = np.vstack([hm_banner, heatmap_blended])
    hm_out_path = out_dir / f"{stem}_caries_heatmap.jpg"
    cv2.imwrite(str(hm_out_path), heatmap_final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f" -> Saved Probability Heatmap: {hm_out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Dental Caries Dual-Stage Prediction on Panoramic X-Rays")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to a single dental panoramic image (.png/.jpg)")
    parser.add_argument("--input-dir", "-d", type=str, default=None, help="Path to a directory containing panoramic images")
    parser.add_argument("--out-dir", "-o", type=str, default=str(BASE_DIR / "visual_results"), help="Directory to save output visualizations")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to model weights pickle bundle (.pkl)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on ('cuda', 'cpu')")
    parser.add_argument("--seg-thr", type=float, default=None, help="Custom threshold for caries mask (default 0.70)")
    args = parser.parse_args()

    # 1. Determine Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using {device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # 2. Locate Model
    model_path = args.model if args.model else find_model_path()
    print(f"[Model] Loading bundle: {model_path}")
    predictor = DualStagePredictor.from_pickle(model_path, device=device)
    print(f"[Model] Successfully initialized DualStagePredictor.")

    # 3. Create Output Directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Gather Target Images
    img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images_to_run = []

    if args.image:
        images_to_run.append(Path(args.image))
    elif args.input_dir:
        input_path = Path(args.input_dir)
        if input_path.is_dir():
            images_to_run.extend([p for p in input_path.iterdir() if p.suffix.lower() in img_extensions])
        else:
            print(f"[Error] Directory not found: {input_path}")
            return
    else:
        # Default behavior: look for images in sample folders or current directory
        search_dirs = [
            BASE_DIR / "visual_results",
            BASE_DIR,
        ]
        for sdir in search_dirs:
            if sdir.is_dir():
                for p in sdir.glob("*.*"):
                    if p.suffix.lower() in img_extensions and "detection" not in p.name and "heatmap" not in p.name:
                        images_to_run.append(p)
                if images_to_run:
                    break

    if not images_to_run:
        print("\nNo images specified or found to test!")
        print("Usage:")
        print("  python test.py --image path/to/your_panoramic_xray.png")
        print("  python test.py --input-dir path/to/images_folder")
        return

    print(f"\nFound {len(images_to_run)} image(s) to process.")
    for img_path in images_to_run:
        process_image(predictor, img_path, out_dir, seg_thr=args.seg_thr)

    print(f"\n=======================================================")
    print(f"All predictions complete! Visualizations saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
