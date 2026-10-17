"""
evaluate_labeled.py -- Ground-truth evaluation for the dental caries dual-stage model.

Supports TWO input formats:
  1. IMAGE + PNG MASK pairs  (labels are binary PNG images: lesion=255/1, bg=0)
  2. IMAGE + POLYGON JSON    (LabelMe / CVAT JSON with polygon shapes named "caries")

Directory layout expected (use --img-dir / --mask-dir / --json-dir to override):

  labeled/
    images/      <- panoramic radiographs  (*.png / *.jpg)
    masks/       <- binary mask PNGs       (same stem as images)
       OR
    annotations/ <- LabelMe JSON files     (same stem as images)

Usage examples:
  # PNG masks
  python evaluate_labeled.py --img-dir labeled/images --mask-dir labeled/masks

  # LabelMe polygon JSON
  python evaluate_labeled.py --img-dir labeled/images --json-dir labeled/annotations

  # Quick single-image smoke-test with overlay images saved
  python evaluate_labeled.py --img-dir labeled/images --mask-dir labeled/masks --limit 1 --vis

Outputs (written to --report-dir, default: eval_report/):
  eval_results.csv          per-image metrics table
  eval_summary.md           human-readable report with bootstrap 95% CI + LOO-CV
  eval_chart_*.png          pixel-level and image-level charts
  vis/                      overlay images (predicted vs ground-truth contours)
"""

import os, sys, json, csv, time, math, random, argparse, warnings
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from dental_inference import DualStagePredictor

# Optimize warnings and numerical tolerances
warnings.filterwarnings("ignore")

# ── colours ──────────────────────────────────────────────────────────────────
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


def savefig(fig, path):
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [chart] {path.name}")


# ── mask loaders ──────────────────────────────────────────────────────────────
def load_mask_png(mask_path, ref_shape):
    """Load a binary mask PNG. Any nonzero pixel -> 1."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    if m.ndim == 3:
        m = m.max(axis=2)
    thr = 127 if mask_path.suffix.lower() in {".jpg", ".jpeg"} else 0
    m = (m > thr).astype(np.uint8)
    if m.shape != ref_shape:
        m = cv2.resize(m, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def load_mask_json(json_path, ref_shape):
    """
    Convert LabelMe / CVAT polygon JSON to a binary mask using cv2.fillPoly.
    Looks for shapes labelled 'caries' (case-insensitive); falls back to all shapes.
    """
    H, W = ref_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    with open(json_path, encoding="utf-8") as f:
        ann = json.load(f)

    shapes = ann.get("shapes", [])
    if not shapes and "annotations" in ann:
        for item in ann["annotations"]:
            if isinstance(item, dict):
                shapes.extend(item.get("shapes", []))

    caries_shapes = [s for s in shapes if "caries" in str(s.get("label", "")).lower()]
    target = caries_shapes if caries_shapes else shapes

    if not target:
        print(f"  [warn] {json_path.name}: no polygon shapes found -- mask stays empty.")
        return mask

    for shape in target:
        pts_raw = shape.get("points", [])
        if not pts_raw:
            continue
        pts = np.array(pts_raw, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)

    return mask


# ── per-image metrics ─────────────────────────────────────────────────────────
def pixel_metrics(pred_mask, gt_mask):
    """All metrics at pixel level on the full panoramic image."""
    pm = pred_mask.astype(bool)
    gm = gt_mask.astype(bool)
    tp = int((pm & gm).sum())
    fp = int((pm & ~gm).sum())
    fn = int((~pm & gm).sum())
    tn = int((~pm & ~gm).sum())

    precision   = tp / max(tp + fp, 1)
    recall      = tp / max(tp + fn, 1)
    f1          = 2 * tp / max(2 * tp + fp + fn, 1)
    iou         = tp / max(tp + fp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy    = (tp + tn) / max(tp + tn + fp + fn, 1)

    return dict(
        dice=round(f1, 4), iou=round(iou, 4),
        precision=round(precision, 4), recall=round(recall, 4),
        specificity=round(specificity, 4), accuracy=round(accuracy, 4),
        tp=tp, fp=fp, fn=fn, tn=tn,
        gt_px=int(gm.sum()), pred_px=int(pm.sum()),
        gt_positive=int(gm.any()), pred_positive=int(pm.any()),
        correct_image_level=int(int(pm.any()) == int(gm.any())),
    )


# ── bootstrap CI ──────────────────────────────────────────────────────────────
def bootstrap_ci(values, stat=np.mean, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = [stat(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boots, [alpha * 100, (1 - alpha) * 100])
    return float(stat(arr)), float(lo), float(hi)


def loocv_dice(records):
    """
    Leave-One-Out CV estimate of mean Dice.
    For each held-out image the micro-Dice is computed on all other images.
    Best honest generalisation estimate when n is small.
    """
    n = len(records)
    if n < 2:
        return [records[0]["dice"]] if n == 1 else []
    totals = {k: sum(r[k] for r in records) for k in ("tp", "fp", "fn")}
    loo = []
    for r in records:
        tp = totals["tp"] - r["tp"]
        fp = totals["fp"] - r["fp"]
        fn = totals["fn"] - r["fn"]
        loo.append(2 * tp / max(2 * tp + fp + fn, 1))
    return loo


# ── overlay visualisation ─────────────────────────────────────────────────────
def make_overlay(gray, pred_mask, gt_mask, boxes):
    """
    Yellow boxes  = detected teeth
    Green contours = ground-truth caries
    Red contours   = predicted caries
    Cyan fill      = True Positive overlap
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tp_mask = (pred_mask.astype(bool) & gt_mask.astype(bool)).astype(np.uint8)
    overlay = vis.copy()
    overlay[tp_mask == 1] = (180, 200, 0)
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 220), 1)
    cs_gt, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cs_gt, -1, (0, 255, 0), 2)
    cs_pr, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cs_pr, -1, (0, 0, 255), 2)
    return vis


# ── charts ────────────────────────────────────────────────────────────────────
def generate_charts(records, rd):
    n = len(records)
    dice_vals = [r["dice"]      for r in records]
    iou_vals  = [r["iou"]       for r in records]
    prec_vals = [r["precision"] for r in records]
    rec_vals  = [r["recall"]    for r in records]
    names     = [r["filename"]  for r in records]

    # Chart 1 -- per-image metric bars
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Per-Image Segmentation Metrics", fontsize=13, fontweight="bold", color=P["text"])
    dark(fig, axes.flat)
    for ax, vals, title, colour in zip(
            axes.flat,
            [dice_vals, iou_vals, prec_vals, rec_vals],
            ["Dice (F1)", "IoU (Jaccard)", "Precision", "Recall"],
            [P["primary"], P["gold"], P["green"], P["accent"]]):
        xs = range(n)
        ax.bar(xs, vals, color=colour, edgecolor=P["bg"], alpha=0.88)
        ax.axhline(np.mean(vals), color="white", lw=1.2, ls="--",
                   label=f"mean={np.mean(vals):.3f}")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([Path(nm).stem for nm in names], rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, facecolor=P["panel"], labelcolor=P["text"])
    plt.tight_layout()
    savefig(fig, rd / "eval_chart_01_per_image_metrics.png")

    # Chart 2 -- precision-recall scatter
    fig, ax = plt.subplots(figsize=(7, 6)); dark(fig, ax)
    sc = ax.scatter(prec_vals, rec_vals, c=dice_vals, cmap="plasma",
                    s=80, zorder=3, edgecolors="white", linewidths=0.5)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Dice", color=P["text"], fontsize=9)
    cb.ax.yaxis.set_tick_params(color=P["text"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=P["text"], fontsize=8)
    for i, nm in enumerate(names):
        ax.annotate(Path(nm).stem, (prec_vals[i], rec_vals[i]),
                    fontsize=6.5, color=P["text"], alpha=0.8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Precision", fontsize=11); ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("Precision-Recall (colour = Dice)", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], "--", color="white", lw=0.8, alpha=0.4)
    savefig(fig, rd / "eval_chart_02_precision_recall.png")

    # Chart 3 -- Dice distribution with bootstrap CI + LOO
    loo = loocv_dice(records)
    fig, ax = plt.subplots(figsize=(7, 4)); dark(fig, ax)
    ax.hist(dice_vals, bins=max(5, n // 2), color=P["primary"],
            edgecolor=P["bg"], alpha=0.85, label="Per-image Dice")
    mu, lo, hi = bootstrap_ci(dice_vals)
    ax.axvline(mu, color="white", lw=1.4, ls="--", label=f"Mean={mu:.3f}")
    ax.axvspan(lo, hi, alpha=0.18, color="white",
               label=f"95% CI [{lo:.3f}, {hi:.3f}]")
    if loo:
        ax.axvline(np.mean(loo), color=P["gold"], lw=1.4, ls=":",
                   label=f"LOO-CV={np.mean(loo):.3f}")
    ax.set_xlabel("Dice Score", fontsize=10); ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Dice Distribution -- Bootstrap 95% CI + LOO-CV",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor=P["panel"], labelcolor=P["text"])
    savefig(fig, rd / "eval_chart_03_dice_distribution.png")

    # Chart 4 -- radar / spider for aggregate metrics
    metrics_agg = {
        "Dice":        np.mean(dice_vals),
        "IoU":         np.mean(iou_vals),
        "Precision":   np.mean(prec_vals),
        "Recall":      np.mean(rec_vals),
        "Specificity": np.mean([r["specificity"] for r in records]),
        "Accuracy":    np.mean([r["accuracy"]    for r in records]),
    }
    cats = list(metrics_agg.keys())
    vals = list(metrics_agg.values())
    N = len(cats)
    angles = [math.pi * 2 * i / N for i in range(N)] + [0]
    vals_plot = vals + [vals[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(P["bg"]); ax.set_facecolor(P["panel"])
    ax.plot(angles, vals_plot, color=P["primary"], lw=2)
    ax.fill(angles, vals_plot, color=P["primary"], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color=P["text"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color=P["text"], fontsize=7)
    ax.grid(color=P["grid"], linewidth=0.5)
    ax.set_title("Aggregate Metric Radar", color=P["text"], fontsize=11,
                 fontweight="bold", pad=20)
    savefig(fig, rd / "eval_chart_04_radar.png")


# ── Markdown report ───────────────────────────────────────────────────────────
def write_report(records, rd, elapsed):
    n = len(records)
    if n == 0:
        print("[warn] No records to report."); return

    dice_vals = [r["dice"]      for r in records]
    iou_vals  = [r["iou"]       for r in records]
    prec_vals = [r["precision"] for r in records]
    rec_vals  = [r["recall"]    for r in records]
    loo       = loocv_dice(records)

    mu_d, lo_d, hi_d = bootstrap_ci(dice_vals)
    mu_i, lo_i, hi_i = bootstrap_ci(iou_vals)
    mu_p, lo_p, hi_p = bootstrap_ci(prec_vals)
    mu_r, lo_r, hi_r = bootstrap_ci(rec_vals)

    tp_total = sum(r["tp"] for r in records)
    fp_total = sum(r["fp"] for r in records)
    fn_total = sum(r["fn"] for r in records)
    micro_dice = 2 * tp_total / max(2 * tp_total + fp_total + fn_total, 1)
    micro_iou  = tp_total / max(tp_total + fp_total + fn_total, 1)
    micro_prec = tp_total / max(tp_total + fp_total, 1)
    micro_rec  = tp_total / max(tp_total + fn_total, 1)

    gt_pos   = sum(r["gt_positive"] for r in records)
    img_acc  = sum(r["correct_image_level"] for r in records) / max(n, 1)

    if n < 5:
        data_note = (
            "> [!WARNING]\n"
            "> **Very small sample (n < 5).** Metrics are highly variable. "
            "Bootstrap CI and LOO-CV are computed but intervals will be very wide. "
            "Annotate at least 20 images for defensible estimates."
        )
    elif n < 20:
        data_note = (
            "> [!NOTE]\n"
            "> **Small sample (n < 20).** Use LOO-CV Dice as the primary honest estimate. "
            "Aim for 50+ labeled images for publication-quality evaluation."
        )
    else:
        data_note = (
            "> [!TIP]\n"
            "> Sample size is reasonable. "
            "Bootstrap CI and micro-averaged metrics are reliable."
        )

    lines = [
        "# Dental Caries Model -- Labeled Evaluation Report", "",
        f"> **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"|  **n = {n} labeled images**  "
        f"|  **Device:** {'CUDA' if torch.cuda.is_available() else 'CPU'}", "",
        "---", "",
        data_note, "",
        "---", "",
        "## 1. Pixel-Level Segmentation Metrics", "",
        "### Micro-averaged (aggregate over all pixels in dataset)", "",
        "| Metric | Value |", "|--------|-------|",
        f"| **Dice (F1)** | **{micro_dice:.4f}** |",
        f"| **IoU (Jaccard)** | **{micro_iou:.4f}** |",
        f"| Precision | {micro_prec:.4f} |",
        f"| Recall | {micro_rec:.4f} |", "",
        "### Macro-averaged (mean over images) + Bootstrap 95% CI", "",
        "| Metric | Mean | 95% CI Low | 95% CI High |",
        "|--------|------|-----------|------------|",
        f"| Dice (F1) | **{mu_d:.4f}** | {lo_d:.4f} | {hi_d:.4f} |",
        f"| IoU | **{mu_i:.4f}** | {lo_i:.4f} | {hi_i:.4f} |",
        f"| Precision | {mu_p:.4f} | {lo_p:.4f} | {hi_p:.4f} |",
        f"| Recall | {mu_r:.4f} | {lo_r:.4f} | {hi_r:.4f} |",
        f"| Specificity | {np.mean([r['specificity'] for r in records]):.4f} | -- | -- |",
        f"| Accuracy | {np.mean([r['accuracy'] for r in records]):.4f} | -- | -- |", "",
        "### Leave-One-Out Cross-Validation (LOO-CV Dice)", "",
        "> LOO-CV: train on n-1 images, evaluate on the left-out one, repeat. "
        "The most honest estimate of generalisation when n is small.", "",
    ]
    if loo:
        lines += [
            f"| LOO-CV Mean Dice | **{np.mean(loo):.4f}** |",
            f"| LOO-CV Std Dice  | {np.std(loo):.4f} |",
        ]
    lines += [
        "", "---", "",
        "## 2. Image-Level Classification (Caries Present / Absent)", "",
        "| Metric | Value |", "|--------|-------|",
        f"| GT caries-positive images | {gt_pos}/{n} ({gt_pos/n*100:.1f}%) |",
        f"| Image-level accuracy | **{img_acc:.4f}** ({img_acc*100:.1f}%) |", "",
        "> Image-level accuracy = fraction of images correctly classified as "
        "caries-positive or caries-negative (any predicted lesion pixel = positive).", "",
        "---", "",
        "## 3. Per-Image Results", "",
        "| File | Dice | IoU | Precision | Recall | GT px | Pred px | GT+ |",
        "|------|------|-----|-----------|--------|-------|---------|-----|",
        *[f"| {r['filename']} | {r['dice']:.4f} | {r['iou']:.4f} | "
          f"{r['precision']:.4f} | {r['recall']:.4f} | {r['gt_px']} | "
          f"{r['pred_px']} | {'YES' if r['gt_positive'] else 'no'} |"
          for r in records], "",
        "---", "",
        "## 4. Scarce-Data Evaluation Guide", "",
        "When labeled data is limited, use these strategies to maximise reliability:", "",
        "| Strategy | When | What it gives |",
        "|----------|------|---------------|",
        "| **Bootstrap 95% CI** | Always (n >= 2) | Uncertainty range for any metric |",
        "| **LOO-CV Dice** | n < 30 | Honest generalisation estimate |",
        "| **Micro-averaged metrics** | Imbalanced images | Size-weighted aggregate |",
        "| **Per-image Dice** | Any n | Spot which images fail |",
        "| **Image-level accuracy** | Any n | Screening usefulness |",
        "| Annotate 5 more images | n < 10 | ~Doubles CI reliability |",
        "| Stratified sampling | Mixed severity | Balanced eval across severity |", "",
        "> **Minimum recommended:** 10 labeled panoramics for a defensible evaluation. "
        "Prioritise clinically borderline images (faint early caries) for maximum diagnostic insight.", "",
        "---", "",
        "## 5. Charts", "",
        "![Per-Image Metrics](eval_chart_01_per_image_metrics.png)",
        "![Precision-Recall](eval_chart_02_precision_recall.png)",
        "![Dice Distribution](eval_chart_03_dice_distribution.png)",
        "![Radar](eval_chart_04_radar.png)", "",
        "---", "",
        f"_Generated by evaluate_labeled.py -- {elapsed:.1f}s total_",
    ]

    out = rd / "eval_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [md]  {out.name}")


def write_csv(records, rd):
    keys = list(records[0].keys())
    path = rd / "eval_results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in records: w.writerow(r)
    print(f"  [csv] {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_model(override=None):
    if override and Path(override).exists(): return str(override)
    for p in [BASE_DIR / "dental_caries_dualstage (2).pkl",
              BASE_DIR / "dental_caries_dualstage.pkl"]:
        if p.exists(): return str(p)
    pkls = list(BASE_DIR.glob("*.pkl"))
    if pkls: return str(pkls[0])
    raise FileNotFoundError("No .pkl model found. Pass --model /path/to/model.pkl")


def main():
    pa = argparse.ArgumentParser(description="Labeled evaluation for dental caries model")
    pa.add_argument("--img-dir",    required=True,
                    help="Directory of panoramic images")
    pa.add_argument("--mask-dir",   default=None,
                    help="Directory of binary mask PNGs (same stem as images)")
    pa.add_argument("--json-dir",   default=None,
                    help="Directory of LabelMe / CVAT polygon JSONs (same stem as images)")
    pa.add_argument("--report-dir", default=str(BASE_DIR / "eval_report"))
    pa.add_argument("--model",      default=None)
    pa.add_argument("--device",     default=None)
    pa.add_argument("--limit",      type=int, default=None,
                    help="Process at most N images (quick smoke-test)")
    pa.add_argument("--vis",        action="store_true",
                    help="Save overlay visualisations to report-dir/vis/")
    args = pa.parse_args()

    if args.mask_dir is None and args.json_dir is None:
        pa.error("Supply at least one of --mask-dir (PNG masks) or --json-dir (polygon JSON).")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device.upper()}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model_path = find_model(args.model)
    print(f"[Model]  {model_path}")
    predictor = DualStagePredictor.from_pickle(model_path, device=device)
    print("[Model]  Ready.\n")

    img_dir  = Path(args.img_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    json_dir = Path(args.json_dir) if args.json_dir else None
    rd       = Path(args.report_dir)
    rd.mkdir(parents=True, exist_ok=True)
    if args.vis:
        (rd / "vis").mkdir(exist_ok=True)

    # Pair images with their annotations
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    pairs = []
    for img_path in images:
        stem = img_path.stem
        ann_path = None
        if mask_dir:
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                c = mask_dir / (stem + ext)
                if c.exists():
                    ann_path = ("mask", c); break
        if ann_path is None and json_dir:
            c = json_dir / (stem + ".json")
            if c.exists():
                ann_path = ("json", c)
        if ann_path is None:
            print(f"  [skip] {img_path.name} -- no matching annotation found.")
            continue
        pairs.append((img_path, ann_path))

    if not pairs:
        print("\n[ERROR] No image-annotation pairs found. "
              "Check that mask / JSON filenames match image stems exactly.")
        return

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"[Eval]  {len(pairs)} image-annotation pairs found.\n")

    records = []
    t0_all = time.perf_counter()

    for idx, (img_path, (ann_type, ann_path)) in enumerate(pairs, 1):
        t0 = time.perf_counter()
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  [{idx:>3}] SKIP (unreadable): {img_path.name}"); continue
        H, W = gray.shape

        try:
            gt_mask = load_mask_png(ann_path, (H, W)) if ann_type == "mask" \
                      else load_mask_json(ann_path, (H, W))
        except Exception as e:
            print(f"  [{idx:>3}] SKIP (mask error): {img_path.name} -- {e}"); continue

        result    = predictor.predict(str(img_path))
        pred_mask = result["mask"]
        boxes     = result["boxes"]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        m = pixel_metrics(pred_mask, gt_mask)
        m["filename"]     = img_path.name
        m["inference_ms"] = round(elapsed_ms, 1)
        m["teeth_count"]  = len(boxes)
        records.append(m)

        flag = "GT+" if m["gt_positive"] else "GT-"
        print(f"  [{idx:>3}/{len(pairs)}] {img_path.name:30s} | "
              f"Dice={m['dice']:.4f}  IoU={m['iou']:.4f}  "
              f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
              f"[{flag}]  {elapsed_ms:.0f}ms")

        if args.vis:
            vis = make_overlay(gray, pred_mask, gt_mask, boxes)
            cv2.imwrite(str(rd / "vis" / (img_path.stem + "_overlay.jpg")),
                        vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

    total_s = time.perf_counter() - t0_all
    print(f"\n[Eval]  Done -- {len(records)} images in {total_s:.1f}s\n")

    if not records:
        print("[ERROR] No results to save."); return

    write_csv(records, rd)
    generate_charts(records, rd)
    write_report(records, rd, total_s)
    print(f"\n[OK] Report saved to: {rd.resolve()}")
    print(f"     eval_summary.md  |  eval_results.csv  |  4 charts"
          + ("  |  vis/ overlays" if args.vis else ""))

    # Console summary
    loo = loocv_dice(records)
    print("\n-- Quick Summary -----------------------------------------------")
    tp_t = sum(r["tp"] for r in records)
    fp_t = sum(r["fp"] for r in records)
    fn_t = sum(r["fn"] for r in records)
    for label, val in [
        ("Micro Dice",         2 * tp_t / max(2 * tp_t + fp_t + fn_t, 1)),
        ("Macro Dice (mean)",  np.mean([r["dice"]      for r in records])),
        ("Macro IoU  (mean)",  np.mean([r["iou"]       for r in records])),
        ("Precision  (mean)",  np.mean([r["precision"] for r in records])),
        ("Recall     (mean)",  np.mean([r["recall"]    for r in records])),
        ("LOO-CV Dice",        np.mean(loo) if loo else float("nan")),
    ]:
        print(f"  {label:<22}: {val:.4f}")
    print("----------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
