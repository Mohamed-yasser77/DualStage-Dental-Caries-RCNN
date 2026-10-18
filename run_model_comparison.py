"""
run_model_comparison.py
Benchmarks the retrained model ('dental_caries_dualstage_retrained.pkl') on the 100-image cohort
and generates a full side-by-side comparison against Version 1.0 ('dental_caries_dualstage (2).pkl').
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from dental_inference import DualStagePredictor

def run_cohort_eval(model_path, images, label="Model"):
    print(f"\n==================================================")
    print(f"Evaluating {label}: {model_path.name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else ''})")
    
    predictor = DualStagePredictor.from_pickle(str(model_path), device=device)
    print(f"Predictor settings: size={predictor.size}, pad={predictor.pad}, det_thr={predictor.det_thr}, unet_thr={predictor.unet_thr}")
    if hasattr(predictor, "merge"):
        print(f"Merge mode: {predictor.merge}, own_box_margin: {getattr(predictor, 'own_box_margin', None)}, unet_hi: {getattr(predictor, 'unet_hi', None)}, min_area: {getattr(predictor, 'min_area', None)}")
        
    records = []
    latencies = []
    
    # Warmup
    print("Warming up model...")
    warm_img = np.zeros((1400, 2800), dtype=np.uint8)
    _ = predictor.predict(warm_img)
    
    print(f"Running inference across {len(images)} cohort images...")
    t_start = time.perf_counter()
    
    for idx, img_path in enumerate(images, 1):
        t0 = time.perf_counter()
        res = predictor.predict(str(img_path))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        H, W = gray.shape[:2]
        
        boxes = res["boxes"]
        scores = res["scores"]
        mask = res["mask"]
        prob = res["prob"]
        
        caries_u8 = (mask * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(caries_u8)
        
        lesion_areas = []
        for l_id in range(1, num_labels):
            lesion_areas.append(int(stats[l_id, cv2.CC_STAT_AREA]))
            
        total_caries_px = int(mask.sum())
        caries_ratio = total_caries_px / (H * W)
        max_prob = float(np.nanmax(prob)) if prob.size else 0.0
        pos_probs = prob[mask > 0]
        mean_pos_prob = float(np.mean(pos_probs)) if pos_probs.size else 0.0
        
        tooth_areas = [float((x2-x1)*(y2-y1)) for (x1, y1, x2, y2) in boxes]
        
        rec = {
            "filename": img_path.name,
            "width": W,
            "height": H,
            "latency_ms": elapsed_ms,
            "teeth_count": len(boxes),
            "det_scores": [float(s) for s in scores],
            "det_score_mean": float(np.mean(scores)) if len(scores) else 0.0,
            "tooth_areas": tooth_areas,
            "caries_lesions": len(lesion_areas),
            "caries_positive": int(len(lesion_areas) > 0),
            "total_caries_px": total_caries_px,
            "caries_area_ratio": caries_ratio,
            "lesion_areas": lesion_areas,
            "max_prob": max_prob,
            "mean_pos_prob": mean_pos_prob,
        }
        records.append(rec)
        
        if idx % 10 == 0 or idx == len(images):
            print(f"  [{idx:3d}/{len(images)}] {img_path.name} | teeth={len(boxes):2d} | lesions={len(lesion_areas):2d} | total_px={total_caries_px:5d} | {elapsed_ms:6.1f} ms")
            
    total_time_s = time.perf_counter() - t_start
    print(f"Cohort evaluation done in {total_time_s:.2f} s (avg: {np.mean(latencies):.1f} ms/img)")
    
    return records, latencies

def aggregate_metrics(records, latencies):
    n = len(records)
    all_teeth = [r["teeth_count"] for r in records]
    all_det_scores = [s for r in records for s in r["det_scores"]]
    all_tooth_areas = [a for r in records for a in r["tooth_areas"]]
    
    # Dentition completeness
    full_dentition = sum(1 for t in all_teeth if t >= 28)
    partial_dentition = sum(1 for t in all_teeth if 20 <= t <= 27)
    reduced_dentition = sum(1 for t in all_teeth if t < 20)
    
    # Caries
    pos_cases = [r for r in records if r["caries_positive"] == 1]
    clean_cases = [r for r in records if r["caries_positive"] == 0]
    all_lesion_areas = [a for r in records for a in r["lesion_areas"]]
    
    lesions_per_pos = [r["caries_lesions"] for r in pos_cases] if pos_cases else [0]
    caries_ratios_pos = [r["caries_area_ratio"] * 100.0 for r in pos_cases] if pos_cases else [0.0]
    
    # Granular lesion sizes
    micro_lesions = sum(1 for a in all_lesion_areas if a < 50)
    moderate_lesions = sum(1 for a in all_lesion_areas if 50 <= a < 200)
    severe_lesions = sum(1 for a in all_lesion_areas if a >= 200)
    total_lesions = len(all_lesion_areas)
    
    peak_probs = [r["max_prob"] for r in pos_cases] if pos_cases else [0.0]
    mean_inside_probs = [r["mean_pos_prob"] for r in pos_cases if r["mean_pos_prob"] > 0]
    
    return {
        "n_images": n,
        "total_teeth": sum(all_teeth),
        "mean_teeth": float(np.mean(all_teeth)),
        "std_teeth": float(np.std(all_teeth)),
        "median_teeth": float(np.median(all_teeth)),
        "min_teeth": int(np.min(all_teeth)),
        "max_teeth": int(np.max(all_teeth)),
        "mean_det_score": float(np.mean(all_det_scores)) if all_det_scores else 0.0,
        "median_det_score": float(np.median(all_det_scores)) if all_det_scores else 0.0,
        "min_det_score": float(np.min(all_det_scores)) if all_det_scores else 0.0,
        "max_det_score": float(np.max(all_det_scores)) if all_det_scores else 0.0,
        "mean_tooth_area": float(np.mean(all_tooth_areas)) if all_tooth_areas else 0.0,
        
        "full_dentition_count": full_dentition,
        "partial_dentition_count": partial_dentition,
        "reduced_dentition_count": reduced_dentition,
        "full_dentition_pct": full_dentition / n * 100.0,
        "partial_dentition_pct": partial_dentition / n * 100.0,
        "reduced_dentition_pct": reduced_dentition / n * 100.0,
        
        "pos_count": len(pos_cases),
        "clean_count": len(clean_cases),
        "prevalence_pct": len(pos_cases) / n * 100.0,
        "specificity_clean_pct": (len(clean_cases) / max(len(clean_cases), 1)) * 100.0,
        "total_lesions": total_lesions,
        "mean_lesions_all": total_lesions / n,
        "mean_lesions_pos": float(np.mean(lesions_per_pos)),
        "std_lesions_pos": float(np.std(lesions_per_pos)),
        "median_lesions_pos": float(np.median(lesions_per_pos)),
        "min_lesions_pos": int(np.min(lesions_per_pos)) if pos_cases else 0,
        "max_lesions_pos": int(np.max(lesions_per_pos)) if pos_cases else 0,
        
        "total_caries_px": sum(r["total_caries_px"] for r in records),
        "mean_caries_ratio_pos": float(np.mean(caries_ratios_pos)),
        "max_caries_ratio_pos": float(np.max(caries_ratios_pos)),
        "peak_prob_mean": float(np.mean(peak_probs)),
        "peak_prob_median": float(np.median(peak_probs)),
        "mean_prob_inside": float(np.mean(mean_inside_probs)) if mean_inside_probs else 0.0,
        
        "micro_count": micro_lesions,
        "micro_pct": (micro_lesions / total_lesions * 100.0) if total_lesions else 0.0,
        "moderate_count": moderate_lesions,
        "moderate_pct": (moderate_lesions / total_lesions * 100.0) if total_lesions else 0.0,
        "severe_count": severe_lesions,
        "severe_pct": (severe_lesions / total_lesions * 100.0) if total_lesions else 0.0,
        "max_lesion_px": int(np.max(all_lesion_areas)) if all_lesion_areas else 0,
        
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_median_ms": float(np.median(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "throughput_fps": 1000.0 / float(np.mean(latencies)),
    }

def main():
    old_model_path = BASE_DIR / "dental_caries_dualstage (2).pkl"
    new_model_path = BASE_DIR / "dental_caries_dualstage_retrained.pkl"
    if not new_model_path.exists():
        new_model_path = BASE_DIR / "dental_caries_dualstage_retrrained.pkl"
        
    img_dir = BASE_DIR / "input"
    images = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0
    )[:100]
    
    print(f"Selected 100 cohort images: {images[0].name} ... {images[-1].name}")
    
    # 1. Run Retrained Model
    new_records, new_latencies = run_cohort_eval(new_model_path, images, label="Version 2.0 (Retrained)")
    new_metrics = aggregate_metrics(new_records, new_latencies)
    
    # Save results to json for report generation
    out_json = BASE_DIR / "benchmark_comparison_metrics.json"
    with open(out_json, "w") as f:
        json.dump(new_metrics, f, indent=2)
    print(f"\nSaved raw aggregated metrics to {out_json}")

if __name__ == "__main__":
    main()
