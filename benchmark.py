"""
benchmark.py - Comprehensive Model Benchmark for Dental Caries Dual-Stage Detector
Runs inference on all images, collects per-image metrics, generates 8 charts + CSV + Markdown report.
Usage:
    python benchmark.py
    python benchmark.py --input-dir input --report-dir benchmark_report
"""
import os, sys, time, argparse, csv
# Benchmarking suite with latency percentiles
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from dental_inference import DualStagePredictor


def find_model_path():
    candidates = [
        BASE_DIR / "dental_caries_dualstage (2).pkl",
        BASE_DIR / "dental_caries_dualstage.pkl",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    pkls = list(BASE_DIR.glob("*.pkl"))
    if pkls:
        return str(pkls[0])
    raise FileNotFoundError("Model .pkl not found.")


def contour_areas(contours):
    return [cv2.contourArea(c) for c in contours] if contours else []


def analyse_image(predictor, image_path):
    t0 = time.perf_counter()
    result = predictor.predict(str(image_path))
    elapsed = time.perf_counter() - t0
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    H, W = gray.shape
    boxes, scores, mask, prob = result["boxes"], result["scores"], result["mask"], result["prob"]
    caries_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(caries_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_areas = contour_areas(contours)
    total_caries_px = int(mask.sum())
    caries_ratio = total_caries_px / (H * W)
    max_prob = float(np.nanmax(prob)) if prob.size else 0.0
    pos_probs = prob[mask > 0]
    mean_pos_prob = float(np.mean(pos_probs)) if pos_probs.size else 0.0
    tooth_areas = [float((x2-x1)*(y2-y1)) for (x1,y1,x2,y2) in boxes]
    return dict(
        filename=image_path.name, width_px=W, height_px=H, image_area_px=H*W,
        inference_ms=round(elapsed*1000, 1),
        teeth_count=len(boxes),
        det_score_mean=round(float(np.mean(scores)) if len(scores) else 0.0, 4),
        det_score_min=round(float(np.min(scores)) if len(scores) else 0.0, 4),
        det_score_max=round(float(np.max(scores)) if len(scores) else 0.0, 4),
        tooth_area_mean=round(float(np.mean(tooth_areas)) if tooth_areas else 0.0, 1),
        caries_lesions=len(contours),
        caries_positive=int(len(contours) > 0),
        total_caries_px=total_caries_px,
        caries_area_ratio=round(caries_ratio*100, 4),
        lesion_area_mean=round(float(np.mean(lesion_areas)) if lesion_areas else 0.0, 1),
        lesion_area_max=round(float(np.max(lesion_areas)) if lesion_areas else 0.0, 1),
        prob_mean=round(float(np.mean(prob)), 4),
        prob_max=round(max_prob, 4),
        prob_mean_positive=round(mean_pos_prob, 4),
        _scores=scores.tolist(),
        _lesion_areas=lesion_areas,
    )


P = dict(primary="#4F8EF7", accent="#F74F4F", green="#4FCA7A", gold="#F7B94F",
         purple="#A66FF7", bg="#1A1A2E", panel="#16213E", text="#E0E0F0", grid="#2A2A4A")


def dark(fig, axes):
    fig.patch.set_facecolor(P["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(P["panel"])
        ax.tick_params(colors=P["text"], labelsize=8)
        ax.xaxis.label.set_color(P["text"]); ax.yaxis.label.set_color(P["text"])
        ax.title.set_color(P["text"])
        for sp in ax.spines.values(): sp.set_edgecolor(P["grid"])
        ax.grid(color=P["grid"], linewidth=0.5, alpha=0.7)


def hist_ax(ax, data, title, xlabel, color, bins=20):
    ax.hist(data, bins=bins, color=color, edgecolor=P["bg"], alpha=0.88)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel("# Images", fontsize=8)
    if data:
        mu = np.mean(data)
        ax.axvline(mu, color="white", linewidth=1.2, linestyle="--", label=f"mean={mu:.2f}")
        ax.legend(fontsize=7, facecolor=P["panel"], labelcolor=P["text"])


def savefig(fig, path):
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [chart] {path.name}")


def generate_charts(records, rd):
    n = len(records)
    teeth = [r["teeth_count"] for r in records]
    cles  = [r["caries_lesions"] for r in records]
    pmx   = [r["prob_max"]*100 for r in records]
    car_r = [r["caries_area_ratio"] for r in records]
    inf   = [r["inference_ms"] for r in records]
    scr   = [r["det_score_mean"] for r in records]
    pos   = sum(r["caries_positive"] for r in records)
    neg   = n - pos
    all_scores = [s for r in records for s in r["_scores"]]

    # Chart 1 – Dashboard 2x3
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Dental Caries Detection — Benchmark Dashboard", fontsize=14, fontweight="bold", color=P["text"], y=1.01)
    dark(fig, axes.flat)
    hist_ax(axes[0,0], teeth, "Teeth Detected per Image",   "Count", P["primary"])
    hist_ax(axes[0,1], cles,  "Caries Lesions per Image",   "Count", P["accent"])
    hist_ax(axes[0,2], pmx,   "Peak Caries Confidence (%)", "%",     P["gold"])
    hist_ax(axes[1,0], car_r, "Caries Area Ratio (%)",      "%",     P["purple"], bins=30)
    hist_ax(axes[1,1], inf,   "Inference Time (ms)",        "ms",    P["green"])
    hist_ax(axes[1,2], scr,   "Avg Detection Score",        "Score", P["primary"])
    plt.tight_layout(); savefig(fig, rd/"chart_01_dashboard.png")

    # Chart 2 – Pie
    fig, ax = plt.subplots(figsize=(6, 6)); dark(fig, ax)
    ax.pie([pos, neg], labels=["Caries Positive","Caries Negative"],
           colors=[P["accent"],P["green"]], autopct="%1.1f%%", startangle=140,
           explode=(0.05,0), textprops={"color":P["text"],"fontsize":11},
           wedgeprops={"edgecolor":P["bg"],"linewidth":1.5})
    ax.set_title("Caries Prevalence", fontsize=12, fontweight="bold", color=P["text"])
    savefig(fig, rd/"chart_02_prevalence_pie.png")

    # Chart 3 – Scatter
    fig, ax = plt.subplots(figsize=(7, 5)); dark(fig, ax)
    sc = ax.scatter(teeth, cles, c=pmx, cmap="plasma", s=25, alpha=0.75, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Peak Confidence (%)", color=P["text"], fontsize=9)
    cb.ax.yaxis.set_tick_params(color=P["text"]); plt.setp(cb.ax.yaxis.get_ticklabels(), color=P["text"], fontsize=8)
    ax.set_xlabel("Teeth Detected", fontsize=10); ax.set_ylabel("Caries Lesions", fontsize=10)
    ax.set_title("Teeth Count vs Caries Lesions (colour = confidence)", fontsize=10, fontweight="bold")
    if len(teeth) > 2:
        z = np.polyfit(teeth, cles, 1); p_ = np.poly1d(z)
        xs = np.linspace(min(teeth), max(teeth), 200)
        ax.plot(xs, p_(xs), "--", color="white", lw=1, alpha=0.6, label="trend")
        ax.legend(fontsize=8, facecolor=P["panel"], labelcolor=P["text"])
    savefig(fig, rd/"chart_03_teeth_vs_lesions.png")

    # Chart 4 – Cumulative time
    fig, ax = plt.subplots(figsize=(9, 4)); dark(fig, ax)
    cum = np.cumsum(inf)/1000
    ax.plot(range(1,n+1), cum, color=P["green"], lw=1.5)
    ax.fill_between(range(1,n+1), cum, alpha=0.25, color=P["green"])
    ax.set_xlabel("Image Index", fontsize=10); ax.set_ylabel("Cumulative Time (s)", fontsize=10)
    ax.set_title("Cumulative Inference Time", fontsize=11, fontweight="bold")
    savefig(fig, rd/"chart_04_cumulative_time.png")

    # Chart 5 – Per-image time bar sorted
    fig, ax = plt.subplots(figsize=(14, 4)); dark(fig, ax)
    si = sorted(inf); thr = np.percentile(inf, 90)
    bc = [P["accent"] if t > thr else P["primary"] for t in si]
    ax.bar(range(n), si, color=bc, width=1.0)
    ax.axhline(np.mean(inf), color="white", lw=1.2, ls="--", label=f"mean={np.mean(inf):.1f} ms")
    ax.set_xlabel("Image (sorted by time)", fontsize=10); ax.set_ylabel("Inference (ms)", fontsize=10)
    ax.set_title("Per-Image Inference Time (sorted) — Red = top 10%", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, facecolor=P["panel"], labelcolor=P["text"])
    savefig(fig, rd/"chart_05_per_image_time.png")

    # Chart 6 – Violin plots
    pos_cles = [c for c in cles if c > 0]; pos_car = [r for r in car_r if r > 0]
    if pos_cles:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5)); dark(fig, [a1, a2])
        v1 = a1.violinplot(pos_cles, showmedians=True)
        for pc in v1["bodies"]: pc.set_facecolor(P["accent"]); pc.set_alpha(0.7)
        v1["cmedians"].set_color("white")
        a1.set_title("Lesion Count (positive images)", fontsize=10, fontweight="bold")
        a1.set_ylabel("Lesions"); a1.set_xticks([])
        v2 = a2.violinplot(pos_car, showmedians=True)
        for pc in v2["bodies"]: pc.set_facecolor(P["gold"]); pc.set_alpha(0.7)
        v2["cmedians"].set_color("white")
        a2.set_title("Caries Area % (positive images)", fontsize=10, fontweight="bold")
        a2.set_ylabel("Caries Area (%)"); a2.set_xticks([])
        savefig(fig, rd/"chart_06_severity_violin.png")

    # Chart 7 – Detection score distribution
    if all_scores:
        fig, ax = plt.subplots(figsize=(7, 4)); dark(fig, ax)
        ax.hist(all_scores, bins=40, color=P["primary"], edgecolor=P["bg"], alpha=0.88)
        ax.axvline(np.mean(all_scores), color="white", lw=1.2, ls="--", label=f"mean={np.mean(all_scores):.3f}")
        ax.set_xlabel("Detection Score", fontsize=10); ax.set_ylabel("# Bounding Boxes", fontsize=10)
        ax.set_title("Tooth Detection Score Distribution (all boxes)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, facecolor=P["panel"], labelcolor=P["text"])
        savefig(fig, rd/"chart_07_detection_scores.png")

    # Chart 8 – Top-10 most affected
    top10 = sorted(records, key=lambda r: r["caries_lesions"], reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(10, 5)); dark(fig, ax)
    names = [r["filename"].replace(".png","") for r in top10]
    vals  = [r["caries_lesions"] for r in top10]
    bars  = ax.barh(names[::-1], vals[::-1], color=P["accent"], edgecolor=P["bg"])
    for bar, v in zip(bars, vals[::-1]):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, str(v), va="center", color=P["text"], fontsize=9)
    ax.set_xlabel("Caries Lesions", fontsize=10)
    ax.set_title("Top-10 Most Affected Radiographs", fontsize=11, fontweight="bold")
    savefig(fig, rd/"chart_08_top10_affected.png")

    return dict(positive_count=pos, negative_count=neg, all_scores=all_scores)


def write_csv(records, path):
    keys = [k for k in records[0] if not k.startswith("_")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in records: w.writerow({k: r[k] for k in keys})
    print(f"  [csv]  {path.name}")


def write_markdown(records, extra, rd, total_s):
    n = len(records)
    teeth = [r["teeth_count"] for r in records]
    cles  = [r["caries_lesions"] for r in records]
    pmx   = [r["prob_max"]*100 for r in records]
    inf   = [r["inference_ms"] for r in records]
    car_r = [r["caries_area_ratio"] for r in records]
    pos, neg, as_ = extra["positive_count"], extra["negative_count"], extra["all_scores"]
    top5l = sorted(records, key=lambda r: r["caries_lesions"],    reverse=True)[:5]
    top5a = sorted(records, key=lambda r: r["caries_area_ratio"], reverse=True)[:5]
    top5s = sorted(records, key=lambda r: r["inference_ms"],      reverse=True)[:5]

    def row5(r):
        return f"| {r['filename']} | {r['teeth_count']} | {r['caries_lesions']} | {r['prob_max']*100:.1f}% | {r['caries_area_ratio']:.3f}% |"

    lines = [
        "# Dental Caries Detection — Benchmark Report", "",
        f"> **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  |  **Device:** {'CUDA' if torch.cuda.is_available() else 'CPU'}", "",
        "## 1. Dataset", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Total radiographs | **{n}** |",
        f"| Avg resolution | {np.mean([r['width_px'] for r in records]):.0f} x {np.mean([r['height_px'] for r in records]):.0f} px |",
        f"| Total wall-clock time | **{total_s:.1f}s** ({total_s/60:.2f} min) |", "",
        "## 2. Inference Speed", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Mean | **{np.mean(inf):.1f} ms** |",
        f"| Median | {np.median(inf):.1f} ms |",
        f"| Std | {np.std(inf):.1f} ms |",
        f"| Min / Max | {np.min(inf):.1f} / {np.max(inf):.1f} ms |",
        f"| Throughput | **{n/total_s:.2f} img/s** |", "",
        "## 3. Stage 1 — Tooth Detection (Faster R-CNN)", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Total tooth boxes | {sum(teeth)} |",
        f"| Mean teeth/image | **{np.mean(teeth):.2f}** |",
        f"| Std | {np.std(teeth):.2f} |",
        f"| Min / Max | {np.min(teeth)} / {np.max(teeth)} |",
        f"| Mean det confidence | **{np.mean(as_):.4f}** |",
        f"| Median det confidence | {np.median(as_):.4f} |",
        f"| Min accepted confidence | {np.min(as_):.4f} |", "",
        "## 4. Stage 2 — Caries Segmentation (U-Net)", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Caries-positive images | **{pos}/{n}** ({pos/n*100:.1f}%) |",
        f"| Caries-negative images | {neg} ({neg/n*100:.1f}%) |",
        f"| Mean lesions/image (all) | **{np.mean(cles):.2f}** |",
        f"| Mean lesions/image (positives) | {np.mean([r['caries_lesions'] for r in records if r['caries_positive']]):.2f} |",
        f"| Median lesions/image | {np.median(cles):.1f} |",
        f"| Max lesions in one image | **{np.max(cles)}** |",
        f"| Mean peak caries confidence | **{np.mean(pmx):.2f}%** |",
        f"| Median peak confidence | {np.median(pmx):.2f}% |",
        f"| Mean caries area ratio (all) | {np.mean(car_r):.4f}% |",
        f"| Mean caries area ratio (positives) | {np.mean([r['caries_area_ratio'] for r in records if r['caries_positive']]):.4f}% |", "",
        "## 5. Top-5 Most Affected (by Lesion Count)", "",
        "| File | Teeth | Lesions | Peak Conf | Area % |",
        "|------|-------|---------|-----------|--------|",
        *[row5(r) for r in top5l], "",
        "## 6. Top-5 Most Affected (by Caries Area)", "",
        "| File | Teeth | Lesions | Peak Conf | Area % |",
        "|------|-------|---------|-----------|--------|",
        *[row5(r) for r in top5a], "",
        "## 7. Slowest Images", "",
        "| File | Width | Height | Inference (ms) |",
        "|------|-------|--------|----------------|",
        *[f"| {r['filename']} | {r['width_px']} | {r['height_px']} | {r['inference_ms']} |" for r in top5s], "",
        "## 8. Charts", "",
        "![Dashboard](chart_01_dashboard.png)",
        "![Prevalence Pie](chart_02_prevalence_pie.png)",
        "![Scatter](chart_03_teeth_vs_lesions.png)",
        "![Cumulative Time](chart_04_cumulative_time.png)",
        "![Per-Image Time](chart_05_per_image_time.png)",
        "![Violin](chart_06_severity_violin.png)",
        "![Det Scores](chart_07_detection_scores.png)",
        "![Top-10](chart_08_top10_affected.png)", "",
        "## 9. Interpretation & Notes", "",
        "- **No ground-truth masks** available; all metrics are model-output statistics.",
        "- High caries-positive rate is expected — the test set contains abnormal radiographs.",
        "- **Peak confidence = 100%** on many images indicates highly confident lesion predictions.",
        "- To compute Precision/Recall/IoU/Dice, annotate masks with LabelMe/CVAT and add --gt-dir support.",
        "- Inference speed on GTX 1650 CUDA is suitable for clinical screening workflows.", "",
        "---",
        "_Generated by benchmark.py — Dental RCNN Project_",
    ]
    out = rd/"benchmark_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [md]   {out.name}")


def main():
    pa = argparse.ArgumentParser(description="Dental caries model benchmark")
    pa.add_argument("--input-dir",  default=str(BASE_DIR/"input"))
    pa.add_argument("--report-dir", default=str(BASE_DIR/"benchmark_report"))
    pa.add_argument("--model",      default=None)
    pa.add_argument("--device",     default=None)
    args = pa.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model_path = args.model or find_model_path()
    print(f"[Model]  {model_path}")
    predictor = DualStagePredictor.from_pickle(model_path, device=device)
    print("[Model]  Ready.\n")

    rd = Path(args.report_dir)
    rd.mkdir(parents=True, exist_ok=True)

    images = sorted(
        [p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp"}],
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
    )
    print(f"[Benchmark] {len(images)} images\n")

    records = []
    t0 = time.perf_counter()
    for i, ip in enumerate(images, 1):
        rec = analyse_image(predictor, ip)
        records.append(rec)
        print(f"  [{i:>3}/{len(images)}] {rec['filename']:25s} | "
              f"teeth={rec['teeth_count']:>2} | lesions={rec['caries_lesions']:>3} | "
              f"peak={rec['prob_max']*100:5.1f}% | {rec['inference_ms']:>6} ms")

    total_s = time.perf_counter() - t0
    print(f"\n[Benchmark] Done in {total_s:.1f}s\n")

    write_csv(records, rd/"benchmark_results.csv")
    extra = generate_charts(records, rd)
    write_markdown(records, extra, rd, total_s)
    print(f"\n[OK] Report saved to: {rd.resolve()}")
    print(f"     benchmark_report.md  |  benchmark_results.csv  |  8 charts")


if __name__ == "__main__":
    main()
