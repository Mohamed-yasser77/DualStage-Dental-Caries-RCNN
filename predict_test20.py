"""
predict_test20.py
Runs the retrained dual-stage model (dental_caries_dualstage_retrained.pkl)
on 20 test panoramic radiographs.
Outputs:
  - Detection overlays (tooth boxes + caries masks + clinical banner)
  - Continuous probability heatmaps
  - Zoomed side-by-side lesion crops
  - summary.json
  - visual_report.html for easy browser inspection
"""

import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from dental_inference import DualStagePredictor

def main():
    model_path = BASE_DIR / "dental_caries_dualstage_retrained.pkl"
    if not model_path.exists():
        model_path = BASE_DIR / "dental_caries_dualstage_retrrained.pkl"
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else ''})")
    print(f"[Model]  Loading {model_path.name} ...")
    predictor = DualStagePredictor.from_pickle(str(model_path), device=device)
    print(f"[Model]  Ready! det_thr={predictor.det_thr}, unet_thr={predictor.unet_thr}, merge={getattr(predictor, 'merge', 'max')}")

    out_dir = BASE_DIR / "test_20_visual_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "lesion_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    image_names = [f"test_{i}.png" for i in range(20)]
    summary = []
    
    print(f"\nProcessing {len(image_names)} images...")
    t_start = time.perf_counter()

    for idx, name in enumerate(image_names):
        img_path = BASE_DIR / "input" / name
        if not img_path.exists():
            print(f"  [Warn] {name} not found in input/ directory.")
            continue

        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            print(f"  [Error] Unreadable image: {name}")
            continue

        H, W = raw_bgr.shape[:2]

        t0 = time.perf_counter()
        res = predictor.predict(str(img_path))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        boxes = res["boxes"]
        scores = res["scores"]
        mask = res["mask"]
        prob = res["prob"]

        caries_u8 = (mask * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(caries_u8)

        lesions_info = []
        for l_id in range(1, num_labels):
            area = int(stats[l_id, cv2.CC_STAT_AREA])
            lx = int(stats[l_id, cv2.CC_STAT_LEFT])
            ly = int(stats[l_id, cv2.CC_STAT_TOP])
            lw = int(stats[l_id, cv2.CC_STAT_WIDTH])
            lh = int(stats[l_id, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[l_id][0]), float(centroids[l_id][1])

            lesion_mask = (labels == l_id)
            lesion_prob = float(prob[lesion_mask].mean()) if lesion_mask.sum() > 0 else 0.0
            lesion_max_prob = float(prob[lesion_mask].max()) if lesion_mask.sum() > 0 else 0.0

            # Associate lesion with containing tooth bbox
            tooth_idx = -1
            for b_idx, (bx1, by1, bx2, by2) in enumerate(boxes):
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    tooth_idx = b_idx
                    break

            crop_rel_name = None
            # Extract zoomed crop for lesions
            pad_crop = 50
            cy1 = max(0, ly - pad_crop)
            cy2 = min(H, ly + lh + pad_crop)
            cx1 = max(0, lx - pad_crop)
            cx2 = min(W, lx + lw + pad_crop)

            if (cy2 - cy1) > 10 and (cx2 - cx1) > 10:
                crop_orig = raw_bgr[cy1:cy2, cx1:cx2].copy()
                crop_vis = crop_orig.copy()
                crop_submask = caries_u8[cy1:cy2, cx1:cx2]

                red_patch = crop_vis.copy()
                red_patch[crop_submask > 0] = [0, 0, 255]
                crop_vis = cv2.addWeighted(red_patch, 0.45, crop_vis, 0.55, 0)

                cnts, _ = cv2.findContours(crop_submask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(crop_vis, cnts, -1, (0, 0, 255), 2)

                side_by_side = np.hstack([crop_orig, crop_vis])
                crop_filename = f"{name}_lesion_{l_id}_prob{int(lesion_max_prob*100)}.jpg"
                cv2.imwrite(str(crops_dir / crop_filename), side_by_side, [cv2.IMWRITE_JPEG_QUALITY, 95])
                crop_rel_name = f"lesion_crops/{crop_filename}"

            lesion_entry = {
                "lesion_id": l_id,
                "area_px": area,
                "bbox": [lx, ly, lx + lw, ly + lh],
                "centroid": [round(cx, 1), round(cy, 1)],
                "mean_prob": round(lesion_prob, 4),
                "max_prob": round(lesion_max_prob, 4),
                "tooth_idx": tooth_idx,
                "crop_image": crop_rel_name,
            }
            lesions_info.append(lesion_entry)

        # 1. Overlay image: teeth (yellow boxes) + caries (red overlay & contour)
        vis = raw_bgr.copy()
        for (bx1, by1, bx2, by2), sc in zip(boxes, scores):
            bx1, by1, bx2, by2 = map(int, [bx1, by1, bx2, by2])
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 240, 255), 2)
            cv2.putText(vis, f"{sc:.2f}", (bx1, max(by1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)

        red_layer = vis.copy()
        red_layer[mask > 0] = [0, 0, 255]
        vis = cv2.addWeighted(red_layer, 0.45, vis, 0.55, 0)
        contours, _ = cv2.findContours(caries_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

        # Header banner
        banner_h = 75
        banner = np.zeros((banner_h, W, 3), dtype=np.uint8)
        cv2.putText(banner, f"Dental Caries Retrained Model (v2.0): {name} ({W}x{H})", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        max_prob_pct = float(prob.max()) * 100.0 if prob.size else 0.0
        cv2.putText(
            banner,
            f"Teeth: {len(boxes)} | Lesions: {len(lesions_info)} | Caries Pixels: {int(mask.sum())} px | Max Prob: {max_prob_pct:.1f}% | Time: {elapsed_ms:.0f}ms",
            (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 1, cv2.LINE_AA
        )
        vis_final = np.vstack([banner, vis])
        overlay_name = f"{name}_overlay.jpg"
        cv2.imwrite(str(out_dir / overlay_name), vis_final, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 2. Probability Heatmap
        prob_u8 = (np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
        heatmap_blend = cv2.addWeighted(raw_bgr, 0.65, heatmap, 0.35, 0)

        hm_banner = np.zeros((banner_h, W, 3), dtype=np.uint8)
        cv2.putText(hm_banner, f"Caries Probability Heatmap: {name}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hm_banner, "Color Map: Dark Blue (Healthy) -> Green/Yellow (Suspicious) -> Bright Red (High Confidence Caries)", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        hm_final = np.vstack([hm_banner, heatmap_blend])
        heatmap_name = f"{name}_heatmap.jpg"
        cv2.imwrite(str(out_dir / heatmap_name), hm_final, [cv2.IMWRITE_JPEG_QUALITY, 95])

        img_summary = {
            "image": name,
            "width": W,
            "height": H,
            "latency_ms": round(elapsed_ms, 1),
            "num_teeth_detected": len(boxes),
            "num_lesions_detected": len(lesions_info),
            "caries_pixel_area": int(mask.sum()),
            "max_caries_probability": round(float(prob.max()), 4),
            "overlay_file": overlay_name,
            "heatmap_file": heatmap_name,
            "lesions": lesions_info,
        }
        summary.append(img_summary)
        print(f"  [{idx+1:2d}/20] {name:12s} | teeth={len(boxes):2d} | lesions={len(lesions_info):2d} | caries_px={int(mask.sum()):5d} | {elapsed_ms:6.1f} ms")

    total_time = time.perf_counter() - t_start
    print(f"\n[Completed] 20 images processed in {total_time:.2f} s")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate visual HTML viewer
    generate_html_report(summary, out_dir)
    print(f"[Viewer] Visual gallery saved to {out_dir / 'visual_report.html'}")

def generate_html_report(summary, out_dir):
    cards_html = []
    for item in summary:
        has_lesions = item["num_lesions_detected"] > 0
        badge_color = "#f87171" if has_lesions else "#4ade80"
        badge_text = f"{item['num_lesions_detected']} Lesions" if has_lesions else "Healthy / Sound"
        
        crops_markup = ""
        if item["lesions"]:
            crop_items = []
            for l in item["lesions"][:4]:
                if l["crop_image"]:
                    crop_items.append(f"""
                        <div class="crop-box">
                            <img src="{l['crop_image']}" alt="Lesion {l['lesion_id']}">
                            <div class="crop-meta">L#{l['lesion_id']} | {l['area_px']} px | {l['max_prob']*100:.0f}% conf</div>
                        </div>
                    """)
            if crop_items:
                crops_markup = f"""
                <div class="crops-row">
                    <div class="crops-title">Zoomed Lesion Patches (Original ↔ Red Overlay):</div>
                    <div class="crops-grid">{''.join(crop_items)}</div>
                </div>
                """

        card = f"""
        <div class="card" id="card-{item['image']}">
            <div class="card-header">
                <div class="card-title">{item['image']}</div>
                <div class="badge" style="background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color};">
                    {badge_text}
                </div>
            </div>
            <div class="meta-row">
                <span><strong>Teeth Detected:</strong> {item['num_teeth_detected']}</span>
                <span><strong>Lesions:</strong> {item['num_lesions_detected']}</span>
                <span><strong>Caries Area:</strong> {item['caries_pixel_area']} px</span>
                <span><strong>Max Confidence:</strong> {item['max_caries_probability']*100:.1f}%</span>
                <span><strong>Latency:</strong> {item['latency_ms']} ms</span>
            </div>
            <div class="img-container">
                <div class="img-col">
                    <div class="img-label">Detection Overlay (Yellow Boxes + Red Contours)</div>
                    <a href="{item['overlay_file']}" target="_blank">
                        <img src="{item['overlay_file']}" alt="Overlay" loading="lazy">
                    </a>
                </div>
                <div class="img-col">
                    <div class="img-label">Probability Heatmap (JET Gradient)</div>
                    <a href="{item['heatmap_file']}" target="_blank">
                        <img src="{item['heatmap_file']}" alt="Heatmap" loading="lazy">
                    </a>
                </div>
            </div>
            {crops_markup}
        </div>
        """
        cards_html.append(card)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dental Caries Retrained Model (v2.0) — Visual Test Gallery</title>
<style>
    body {{
        margin: 0;
        padding: 24px;
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    .header {{
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 1px solid #30363d;
    }}
    h1 {{
        margin: 0 0 10px 0;
        font-size: 26px;
        color: #58a6ff;
    }}
    .subtitle {{
        color: #8b949e;
        font-size: 14px;
    }}
    .stats-bar {{
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 15px;
    }}
    .stat-pill {{
        background: #161b22;
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid #30363d;
        font-size: 13px;
    }}
    .stat-pill strong {{ color: #f0883e; }}
    .gallery {{
        display: flex;
        flex-direction: column;
        gap: 28px;
        max-width: 1300px;
        margin: 0 auto;
    }}
    .card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }}
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }}
    .card-title {{
        font-size: 18px;
        font-weight: 600;
        color: #f0f6fc;
    }}
    .badge {{
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }}
    .meta-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        font-size: 13px;
        color: #8b949e;
        background: #0d1117;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 16px;
    }}
    .meta-row strong {{ color: #c9d1d9; }}
    .img-container {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }}
    .img-col {{
        display: flex;
        flex-direction: column;
    }}
    .img-label {{
        font-size: 12px;
        color: #8b949e;
        margin-bottom: 6px;
        font-weight: 500;
    }}
    .img-col img {{
        width: 100%;
        border-radius: 6px;
        border: 1px solid #30363d;
        background: #000;
        cursor: pointer;
        transition: transform 0.2s;
    }}
    .img-col img:hover {{
        transform: scale(1.01);
    }}
    .crops-row {{
        margin-top: 16px;
        padding-top: 14px;
        border-top: 1px solid #21262d;
    }}
    .crops-title {{
        font-size: 12px;
        color: #8b949e;
        margin-bottom: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .crops-grid {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }}
    .crop-box {{
        background: #0d1117;
        padding: 6px;
        border-radius: 6px;
        border: 1px solid #30363d;
    }}
    .crop-box img {{
        height: 80px;
        border-radius: 4px;
    }}
    .crop-meta {{
        font-size: 10px;
        color: #8b949e;
        text-align: center;
        margin-top: 4px;
    }}
</style>
</head>
<body>
<div class="header">
    <h1>Dental Caries Dual-Stage System (Version 2.0 Retrained)</h1>
    <div class="subtitle">Visual Detection Overlays & Probability Heatmaps for 20 Panoramic Radiographs (`test_0.png` – `test_19.png`)</div>
    <div class="stats-bar">
        <div class="stat-pill">Model: <strong>dental_caries_dualstage_retrained.pkl</strong></div>
        <div class="stat-pill">Hardware: <strong>NVIDIA GPU (CUDA)</strong></div>
        <div class="stat-pill">Merge Mode: <strong>Weighted Own-Box Region Voting</strong></div>
    </div>
</div>

<div class="gallery">
    {''.join(cards_html)}
</div>
</body>
</html>
"""
    with open(out_dir / "visual_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    main()
