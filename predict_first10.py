import os
import sys
import json
import cv2
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from dental_inference import DualStagePredictor, expand_box

def main():
    model_path = BASE_DIR / "dental_caries_dualstage (2).pkl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    predictor = DualStagePredictor.from_pickle(str(model_path), device=device)
    
    out_dir = BASE_DIR / "test_10_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "lesion_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # First 10 test images numerically test_0.png to test_9.png
    image_names = [f"test_{i}.png" for i in range(10)]
    
    summary = []
    
    for idx, name in enumerate(image_names):
        img_path = BASE_DIR / "input" / name
        if not img_path.exists():
            print(f"Warning: {name} not found!")
            continue
            
        raw_bgr = cv2.imread(str(img_path))
        H, W = raw_bgr.shape[:2]
        gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        
        # Predict
        res = predictor.predict(str(img_path))
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
            
            # Determine which tooth bbox contains this lesion
            tooth_idx = -1
            for b_idx, (bx1, by1, bx2, by2) in enumerate(boxes):
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    tooth_idx = b_idx
                    break
                    
            lesion_entry = {
                "lesion_id": l_id,
                "area_px": area,
                "bbox": [lx, ly, lx + lw, ly + lh],
                "centroid": [round(cx, 1), round(cy, 1)],
                "mean_prob": round(lesion_prob, 4),
                "max_prob": round(lesion_max_prob, 4),
                "tooth_idx": tooth_idx
            }
            lesions_info.append(lesion_entry)
            
            # Save cropped visual of the lesion
            pad_crop = 50
            cy1 = max(0, ly - pad_crop)
            cy2 = min(H, ly + lh + pad_crop)
            cx1 = max(0, lx - pad_crop)
            cx2 = min(W, lx + lw + pad_crop)
            
            crop_orig = raw_bgr[cy1:cy2, cx1:cx2].copy()
            crop_vis = crop_orig.copy()
            crop_mask = caries_u8[cy1:cy2, cx1:cx2]
            
            # overlay red
            red_patch = crop_vis.copy()
            red_patch[crop_mask > 0] = [0, 0, 255]
            crop_vis = cv2.addWeighted(red_patch, 0.5, crop_vis, 0.5, 0)
            
            # draw contours
            cnts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop_vis, cnts, -1, (0, 0, 255), 2)
            
            # save side-by-side
            side_by_side = np.hstack([crop_orig, crop_vis])
            crop_filename = f"{name}_lesion_{l_id}_prob_{int(lesion_max_prob*100)}.jpg"
            cv2.imwrite(str(crops_dir / crop_filename), side_by_side)

        # 1. Overlay image: teeth (yellow) + caries (red)
        vis = raw_bgr.copy()
        for (bx1, by1, bx2, by2), sc in zip(boxes, scores):
            bx1, by1, bx2, by2 = map(int, [bx1, by1, bx2, by2])
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 240, 255), 2)
            cv2.putText(vis, f"{sc:.2f}", (bx1, max(by1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1)
            
        red_layer = vis.copy()
        red_layer[mask > 0] = [0, 0, 255]
        vis = cv2.addWeighted(red_layer, 0.45, vis, 0.55, 0)
        contours, _ = cv2.findContours(caries_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
        
        # Banner
        banner_h = 75
        banner = np.zeros((banner_h, W, 3), dtype=np.uint8)
        cv2.putText(banner, f"Dental Caries Prediction: {name} ({W}x{H})", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(banner, f"Teeth: {len(boxes)} | Lesions: {len(lesions_info)} | Caries Pixels: {int(mask.sum())} | Max Prob: {float(prob.max())*100:.1f}%", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 1)
        vis_final = np.vstack([banner, vis])
        cv2.imwrite(str(out_dir / f"{name}_overlay.jpg"), vis_final)
        
        # 2. Probability Heatmap
        prob_u8 = (np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
        heatmap_blend = cv2.addWeighted(raw_bgr, 0.65, heatmap, 0.35, 0)
        hm_banner = np.zeros((banner_h, W, 3), dtype=np.uint8)
        cv2.putText(hm_banner, f"Caries Probability Heatmap: {name}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(hm_banner, "Blue: Healthy | Green/Yellow: Low/Mid Conf | Red: High Confidence Caries", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        hm_final = np.vstack([hm_banner, heatmap_blend])
        cv2.imwrite(str(out_dir / f"{name}_heatmap.jpg"), hm_final)
        
        img_summary = {
            "image": name,
            "width": W,
            "height": H,
            "num_teeth_detected": len(boxes),
            "num_lesions_detected": len(lesions_info),
            "caries_pixel_area": int(mask.sum()),
            "max_caries_probability": round(float(prob.max()), 4),
            "lesions": lesions_info
        }
        summary.append(img_summary)
        print(f"[{idx+1}/10] {name}: {len(boxes)} teeth, {len(lesions_info)} lesions, max_prob={img_summary['max_caries_probability']}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nCompleted prediction for first 10 images.")

if __name__ == "__main__":
    main()
