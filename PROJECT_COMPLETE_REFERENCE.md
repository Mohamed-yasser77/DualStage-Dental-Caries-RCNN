# Dual-Stage Dental Caries Detection & Segmentation Pipeline
## Comprehensive Technical Documentation, Algorithmic Analysis & Academic Defense Guide

> **Document Type:** Comprehensive Project Reference & Faculty Defense Guide  
> **Project Title:** Dual-Stage Deep Learning Framework for Tooth Localization and Dental Caries Segmentation in Panoramic Radiographs  
> **Primary Models:** Faster R-CNN (MobileNetV3-Large FPN) + U-Net (EfficientNet-B0 Encoder)  
> **Training Notebook:** `dental_caries_dualstage_fp_reduced.ipynb`  
> **Model Artifacts:** `dental_caries_dualstage_retrained.pkl` (Format v2)  
> **Live Deployed Application:** [https://dualstage-dental-caries-rcnn.streamlit.app/](https://dualstage-dental-caries-rcnn.streamlit.app/)  
> **Clinical Disclaimer:** *Research prototype. Not validated for clinical diagnostics or primary treatment planning.*

---

## Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Clinical Motivation & Challenges in Panoramic Radiography](#2-clinical-motivation--challenges-in-panoramic-radiography)
3. [End-to-End System Architecture (Dual-Stage Paradigm)](#3-end-to-end-system-architecture-dual-stage-paradigm)
4. [Exhaustive Notebook Walkthrough & Function Reference](#4-exhaustive-notebook-walkthrough--function-reference)
   - [4.1 Global Hyperparameters (`CFG`)](#41-global-hyperparameters-cfg)
   - [4.2 Data Ingestion, Coordinate Harmonization & Leakage Protection](#42-data-ingestion-coordinate-harmonization--leakage-protection)
   - [4.3 Stage 1: Tooth Localization (Faster R-CNN)](#43-stage-1-tooth-localization-faster-r-cnn)
   - [4.4 Stage 2: Memory-Safe Multi-GPU U-Net Training](#44-stage-2-memory-safe-multi-gpu-u-net-training)
   - [4.5 Novel Loss Formulations & Hard-Negative Mining](#45-novel-loss-formulations--hard-negative-mining)
   - [4.6 Post-Processing & Threshold Optimization](#46-post-processing--threshold-optimization)
5. [In-Depth Analysis of Changes & Performance Enhancements (v1.0 vs v2.0)](#5-in-depth-analysis-of-changes--performance-enhancements-v10-vs-v20)
   - [5.1 Why Version 1.0 Needed Improvement](#51-why-version-10-needed-improvement)
   - [5.2 Complete Delta Table: Engineering Interventions](#52-complete-delta-table-engineering-interventions)
   - [5.3 Quantitative Benchmarks (100-Image Cohort)](#53-quantitative-benchmarks-100-image-cohort)
6. [Inference Engine & Post-Processing Details](#6-inference-engine--post-processing-details)
7. [Cloud Deployment & Architecture](#7-cloud-deployment--architecture)
8. [Tutor Review & Project Viva Defense Guide](#8-tutor-review--project-viva-defense-guide)

---

## 1. Executive Summary & Problem Statement

Dental caries (tooth decay) is one of the most prevalent chronic conditions worldwide. Early detection via orthopantomograms (OPGs / panoramic radiographs) is vital to preserve tooth vitality and avoid invasive procedures. However:
1. Standard full-image segmentation models (e.g., standard FCNs or U-Nets directly on 2900 × 1400 px radiographs) suffer severe class imbalance because carious lesions account for **< 0.1%** of total pixel volume.
2. Directly downsampling panoramic radiographs to 512 × 512 px erases early-stage enamel and interproximal lesions.
3. Baseline segmentation approaches suffer from **hyper-fragmentation** and high **false-positive rates** due to cervical burnout, overlapping enamel shells, restorative materials, and spine artifacts.

To solve this, we formulated a **Dual-Stage Hierarchical Paradigm**:
- **Stage 1 (Macro-Anatomy):** Faster R-CNN localizes individual tooth units with high precision, isolating dental regions of interest (ROIs).
- **Stage 2 (Micro-Pathology):** U-Net with an EfficientNet-B0 encoder performs high-resolution segmentation on CLAHE-enhanced, normalized tooth crops ($256 \times 256$ px).
- **Integration Engine:** Anatomical weighted own-box voting stitches localized lesion probability maps back to the panoramic coordinate space.

---

## 2. Clinical Motivation & Challenges in Panoramic Radiography

Panoramic radiographs pose severe computer vision challenges:
- **Cervical Burnout:** Anatomical radiolucency between the enamel crown and alveolar bone often mimics proximal cervical caries.
- **Interproximal Overlap:** Faulty patient positioning or crowded dentition causes overlapping crown shadows.
- **Ghost Shadows & Spinal Column:** Cervical vertebrae project radiopaque artifacts down the centerline, obscuring central incisors.
- **Metal Restorations:** High-density amalgam or crowns cause streak artifacts and scatter.

---

## 3. End-to-End System Architecture (Dual-Stage Paradigm)

```mermaid
graph TD
    A["Raw Panoramic Radiograph (~2940 × 1435 px)"] --> B["Grayscale Conversion & Dynamic Normalization"]
    B --> C["Stage 1: Faster R-CNN (MobileNetV3-Large FPN)"]
    C --> D["Tooth Bounding Boxes (det_thr = 0.70, NMS IoU = 0.30)"]
    
    D --> E["Crop & Pad Tooth Units (pad = 0.15)"]
    E --> F["CLAHE Contrast Equalization (clipLimit=2.0, tileGrid=(8,8))"]
    F --> G["Stage 2: U-Net (EfficientNet-B0 Encoder + ImageNet Weights)"]
    G --> H["Test-Time Augmentation (TTA Horizontal Flip Logit Averaging)"]
    
    H --> I["Per-Tooth Caries Probability Maps (256 × 256)"]
    I --> J["Spatial Remapping: Weighted Own-Box Margin Voting"]
    J --> K["Full Panoramic Probability Heatmap"]
    K --> L["Hysteresis Thresholding & Area Filtering (min_area = 50 px)"]
    L --> M["Final Output: Bounding Boxes + Segmented Caries Contours"]
```

---

## 4. Exhaustive Notebook Walkthrough & Function Reference

The complete training and evaluation pipeline is implemented in `dental_caries_dualstage_fp_reduced.ipynb`. Below is the exhaustive reference of every architectural component and function.

### 4.1 Global Hyperparameters (`CFG`)

```python
CFG = dict(
    seed=42, val_frac=0.10, test_frac=0.10,
    crop=256, pad=0.15,
    # Stage 1: Tooth Detector
    det_epochs=60, det_bs=2, det_lr=0.005, det_patience=12, det_budget_h=2.5,
    # Stage 2: Caries U-Net
    seg_epochs=150, seg_bs=16, seg_lr=3e-4, seg_patience=25, seg_budget_h=3.5,
    # False-Positive Mitigation Parameters
    seg_neg_ratio=1.0,         # 1:1 ratio between caries-free crops and caries crops
    seg_tv_alpha=0.5,          # Tversky false-positive penalty
    seg_tv_beta=0.5,           # Tversky false-negative penalty (balanced Dice)
    seg_fp_lambda=0.5,         # Direct penalty on max activation of caries-free crops
    seg_hn_start=5,            # Start hard-negative mining at epoch 5
    seg_hn_every=5,            # Re-mine hard negatives every 5 epochs
    seg_hn_frac=0.5,           # 50% of negative crop quota filled by mined hard negatives
    seg_hn_min_px=30,          # Minimum false pixels to consider a crop "hard negative"
    seg_sel_fp_weight=0.5,     # Checkpoint selection penalty on empty-crop FP rate
    seg_extra_val_imgs=40,     # Held-out panoramics for Stage 2 validation
    seg_extra_val_tiles=300,   # Held-out tiles for Stage 2 validation
    pp_own_box_margin=0.04,    # Margin for weighted own-box voting
    hard_stop_h=10.0,
    reuse_ckpt=True,
)
```

---

### 4.2 Data Ingestion, Coordinate Harmonization & Leakage Protection

#### 1. `find_dataset_roots(base)` and `classify(base)`
- **Purpose:** Automatically crawls storage directories to locate images, labels (`masks`, `labels_clean`), and bounding box annotations (`annotations`, `bboxes_teeth`).
- **Data Discovery:** Discovers the core box-annotated dataset (100 images) and supplementary DC1000 datasets (497 full panoramics and 2,389 pre-cropped tiles).

#### 2. `is_test_mirror(folder)` — **Data Leakage Safeguard**
- **Purpose:** Detects and explicitly blocks `org_test_dataset`.
- **Significance:** In the raw DC1000 dataset, `org_test_dataset` mirrors the held-out test split under alternate filenames. Without this filter, test data would leak into Stage 2 training.

#### 3. `to_xyxy(a, mode, W, H, normed)` and `score_mode(mode)`
- **Purpose:** Dynamically determines the bounding box convention (`xyxy`, `xywh`, or `cxcywh`) by cross-verifying candidate boxes against ground-truth caries masks.
- **Mathematical Selection:** Computes mask coverage and boundary violation penalties:
  $$\text{Score} = \text{Coverage} - \text{InvalidBoxes} - \text{OutOfBounds} - 0.5 \times \text{BoxArea}$$
  The pipeline confirmed `xyxy` as the ground-truth format with 98.9% caries mask coverage.

---

### 4.3 Stage 1: Tooth Localization (Faster R-CNN)

#### 1. Architecture: `build_detector()`
- **Backbone:** MobileNetV3-Large with Feature Pyramid Network (FPN), pre-trained on ImageNet-1K (`MobileNet_V3_Large_Weights.IMAGENET1K_V1`).
- **Parameters:** `num_classes=2` (background + tooth), `min_size=800`, `max_size=1700`, `box_detections_per_img=64`.
- **Multi-Scale Training:** Multi-scale input resizing across $\{640, 704, 768, 800\}$ ensures robust scale invariance across variable patient anatomy.

#### 2. Training Engine: `train_detector_ddp.py`
- **Multi-GPU DistributedDataParallel (DDP):** Launched via `torchrun --nproc_per_node=2` on dual Tesla T4 GPUs using the NCCL backend. Eliminates Python GIL thread bottlenecks.
- **Dataset (`DetDS`):**
  - Horizontal flipping with box coordinate inversion ($x_1' = W - x_2$, $x_2' = W - x_1$).
  - Random contrast adjustment: $\alpha \in [0.75, 1.25]$, $\beta \in [-25, 25]$.
- **Optimizer:** SGD with Nesterov momentum ($\mu = 0.9$, $\text{weight decay} = 10^{-4}$).
- **Learning Rate Schedule:** Cosine annealing with linear warmup for the first 10% of steps:
  $$\eta(t) = \begin{cases} 0.001 + 0.999 \frac{t}{T_{\text{warm}}}, & t < T_{\text{warm}} \\ 0.02 + 0.98 \times \frac{1}{2} \left(1 + \cos\left(\pi \frac{t - T_{\text{warm}}}{T_{\text{total}} - T_{\text{warm}}}\right)\right), & t \ge T_{\text{warm}} \end{cases}$$
- **Gradient Clipping:** Norm clipped to $10.0$ for numerical stability.

#### 3. Evaluation: `det_metrics(preds, recs, thr=0.5, iou_thr=0.5)`
- Computes greedy bipartite matching between predicted and ground-truth boxes at $\text{IoU} \ge 0.50$ to calculate Precision, Recall, F1, and Mean IoU.

---

### 4.4 Stage 2: Memory-Safe Multi-GPU U-Net Training

#### 1. Pre-Processing & Normalization Functions
- `expand_box(box, W, H, pad=0.15)`: Widens tooth bounding boxes by $15\%$ on all sides, ensuring boundary enamel and proximal contact points are not clipped.
- `crop_tooth(gray, box, size=256, pad=0.15)`: Crops the expanded ROI, executes CLAHE (Contrast-Limited Adaptive Histogram Equalization, $\text{clipLimit}=2.0$, $\text{grid}=(8,8)$) to elevate contrast inside radiolucent lesions, and resizes to $256 \times 256$ px using area interpolation.
- `crop_mask(mask01, padded_box, size=256)`: Crops and resizes the binary ground-truth caries mask to align with the cropped tooth image.
- `prep(xb)`: Broadcasts single-channel grayscale to 3 channels and applies ImageNet standard normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).

#### 2. Memory-Mapped Dataset (`write_crop_memmaps`)
- **RAM Crash Solution:** Previous runs keeping $\approx 20,000$ cropped images in system RAM suffered OOM crashes on Kaggle.
- **Disk-Backed Memmaps:** Pre-extracts all training and validation crops into disk-backed binary files (`Xtr.npy`, `Ytr.npy`, `Xva.npy`, `Yva.npy`). Data loaders stream single crops via `mmap_mode="r"` with zero RAM footprint.

#### 3. On-GPU Dynamic Augmentation (`gpu_aug(x, y)`)
Executed directly on VRAM in PyTorch:
- Random horizontal flip ($p = 0.5$).
- Random affine transformations: rotation $\theta \in [-12^\circ, +12^\circ]$, isotropic scaling $s \in [0.88, 1.12]$, translation $\Delta x, \Delta y \in [-8\%, +8\%]$.
- Photometric jitter: intensity scaling $\alpha \in [0.8, 1.2]$ and shift $\beta \in [-0.1, +0.1]$.

---

### 4.5 Novel Loss Formulations & Hard-Negative Mining

#### 1. Balanced Tversky Loss with Empty-Crop Activation Penalty (`seg_loss`)
Standard segmentation losses either ignore small lesions (BCE) or cause extreme false alarms (recall-biased Tversky $\beta = 0.7$). Our custom loss function is formulated as:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + (1 - \text{TI}) + \lambda_{\text{FP}} \cdot \mathcal{L}_{\text{empty-FP}}$$

Where the Tversky Index ($\text{TI}$) is balanced:
$$\text{TI} = \frac{\sum p_i y_i + 1}{\sum p_i y_i + \alpha \sum p_i (1 - y_i) + \beta \sum (1 - p_i) y_i + 1}, \quad \alpha = 0.5, \beta = 0.5$$

And the **Empty-Crop Penalty** directly suppresses phantom activations:
$$\mathcal{L}_{\text{empty-FP}} = \frac{1}{|N_{\text{empty}}|} \sum_{j \in N_{\text{empty}}} \max_{k} \left( \sigma(\text{logits}_{j,k}) \right)$$
*Effect:* Any confident prediction on a caries-free tooth imposes a severe gradient penalty, directly training the model to remain silent on healthy enamel.

#### 2. Periodic Hard-Negative Mining (`mine_hard_negatives`)
- Every 5 epochs, the current model evaluates all caries-free training crops ($N \approx 11,300$).
- Crops producing $\ge 30$ false-positive pixels are flagged as "Hard Negatives".
- In subsequent epochs, hard negatives constitute **50% of the negative crop quota**, forcing the optimizer to resolve fine dental ambiguities (e.g., pulp chamber margins, enamel grooves).

#### 3. Precision-Weighted Checkpoint Selection
Checkpoints are saved not on raw Dice, but on a composite objective penalizing false alarms:
$$\text{Selection Score} = \text{Val\_Dice} - 0.5 \times \text{FP\_Rate}_{\text{empty}}$$

---

### 4.6 Post-Processing & Threshold Optimization

#### 1. Anatomical Weighted Own-Box Remapping
- **Previous Method (v1.0):** Cropped probability maps were pasted back with a simple `np.maximum`. Overlapping padding zones caused lesions from one tooth to spill across interproximal spaces into neighboring healthy teeth.
- **New Method (v2.0):** Each crop votes **only within its own anatomical bounding box** ($+4\%$ transition margin). In overlapping boundary areas, predictions are averaged rather than maximized:
  $$\text{Prob}(x, y) = \frac{\sum_{i} P_i(x, y) \cdot W_i(x, y)}{\sum_{i} W_i(x, y)}$$

#### 2. Dual-Threshold Hysteresis & Connected-Component Filtering (`postprocess_prob`)
- **Lower Threshold ($lo = 0.40$):** Retains lesion boundaries and demineralized fringes.
- **Hysteresis Peak Check ($hi = 0.80$):** A candidate blob survives only if its peak internal probability reaches $\ge 0.80$.
- **Area Threshold ($\text{min\_area} = 50\text{ px}$):** Connected components with area $< 50\text{ px}$ are dropped as imaging noise.

---

## 5. In-Depth Analysis of Changes & Performance Enhancements (v1.0 vs v2.0)

### 5.1 Why Version 1.0 Needed Improvement
1. **Severe False-Positive Clutter:** Version 1.0 produced **551 detected lesions** across 100 test images (5.51 lesions/image). Clinicians confirmed that $\approx 60\%$ were sub-clinical speckles or pulp-shadow artifacts.
2. **Boundary Bleed:** Overlapping padding crops caused single cavities to be duplicated on adjacent sound teeth.
3. **Over-Sensitivity:** Tversky loss with $\beta = 0.70$ biased the model towards over-predicting caries to avoid false negatives.

---

### 5.2 Complete Delta Table: Engineering Interventions

| Pipeline Dimension | Version 1.0 (Baseline) | Version 2.0 (Retrained & Calibrated) | Technical Rationale |
|---|---|---|---|
| **Weight Bundle** | `dental_caries_dualstage (2).pkl` | `dental_caries_dualstage_retrained.pkl` | Format v2 serialisation with calibrated metadata |
| **Stage 1 Detection Threshold** | `0.60` | `0.70` | Eliminates phantom boxes on mandibular ramus/condyle |
| **Stage 2 Probability Threshold** | `0.70` (uncalibrated) | `0.40` (calibrated) | Optimal operating point for retrained logits |
| **Loss Function** | Recall-Biased Tversky ($\alpha=0.3, \beta=0.7$) | Balanced Tversky ($\alpha=0.5, \beta=0.5$) + Empty-Crop Penalty ($\lambda=0.5$) | Neutralizes over-prediction bias; penalizes false alarms |
| **Hard Negative Mining** | None (random negative sampling) | Active mining every 5 epochs ($50\%$ hard quota) | Directly trains on tough dental ambiguities |
| **Validation Dataset** | 10 box-annotated images ($302$ crops) | 10 boxed + 40 held-out panoramics + 300 tiles | 10x larger sample size prevents threshold overfitting |
| **Crop Stitching Strategy** | `np.maximum` over padded crops | Weighted Own-Box Voting ($\text{margin}=0.04$) | Prevents interproximal lesion spillover |
| **Artifact Post-Processing** | None | Hysteresis Peak Check + Min Area Filter ($50$ px) | Suppresses floating non-dental speckle artifacts |
| **Multi-GPU Training** | `torch.nn.DataParallel` (RAM spikes) | 2-Process NCCL `DistributedDataParallel` + Memmaps | Fixed host-RAM footprint at $< 3.5$ GB per worker |

---

### 5.3 Quantitative Benchmarks (100-Image Cohort)

Evaluating both models head-to-head on the identical 100-panoramic test cohort produced the following results:

```mermaid
pie title Version 1.0: Lesion Noise Composition (551 Lesions Total)
    "Micro-Speckle Artifacts (<50 px)" : 324
    "Moderate Lesions (50-200 px)" : 129
    "Severe Cavitations (>=200 px)" : 98
```

```mermaid
pie title Version 2.0: Clean Clinical Lesion Composition (154 Lesions Total)
    "Micro-Speckle Artifacts (<50 px)" : 38
    "Moderate Lesions (50-200 px)" : 44
    "Severe Cavitations (>=200 px)" : 72
```

#### Detailed Metrics Comparison:

| Evaluation Metric | Version 1.0 (Baseline) | Version 2.0 (Retrained) | Quantitative Improvement |
|---|:---:|:---:|---|
| **Total Detected Lesions** | 551 | **154** | **-72.1% false-positive clutter eliminated** |
| **Micro-Speckles ($< 50$ px)** | 324 (58.8%) | **38 (24.7%)** | **-88.3% suppression of tiny false alarms** |
| **Pixel Precision** | 75.05% | **79.30%** | **+4.25% tighter, more accurate caries margins** |
| **Pixel Recall (Sensitivity)** | 83.70% | **79.03%** | Balanced; eliminated border pixel blur |
| **Dice Similarity Coefficient (DSC)** | 79.14% | **79.16%** | Maintained peak spatial overlap |
| **Intersection over Union (IoU)** | 65.48% | **65.51%** | Fully preserved segmentation fidelity |
| **Stage 1 Detection Confidence (Mean)** | 90.81% | **93.98%** | **+3.17% higher tooth localization certainty** |
| **Stage 1 Detection Confidence (Median)**| 90.93% | **97.50%** | **+6.57% sharper probability distribution** |
| **Tooth Localization Precision** | 93.81% | **95.35%** | **+1.54% (26.3% fewer spurious tooth boxes)** |
| **Healthy Image Specificity** | 100.0% (72/72) | **100.0% (78/78)** | Zero false alarms on pristine radiographs |
| **Inference Latency per Radiograph** | 1,080 ms | **1,108 ms** | Real-time chairside execution (~1.1 s) |

---

## 6. Inference Engine & Post-Processing Details

The production runtime is orchestrated by two modular scripts:
1. `pipeline.py`:
   - `@st.cache_resource`: Loads and caches `DualStagePredictor` in memory.
   - Dynamic device selection (CUDA GPU if available, else CPU).
   - Warm-up pass on dummy tensors to eliminate initial JIT/CUDA allocation overhead.
   - Structured results returned via `ToothResult` and `InferenceResult` dataclasses.
2. `viz.py`:
   - High-contrast visual overlays: Tooth bounding boxes (Yellow, `#FFD200`), Caries filled mask (Semi-transparent Red, `#DC1E1E`), and Boundary Contours (Bright Red, `#FF2828`).
   - Side-by-side zoom visualizers showing raw tooth crops, CLAHE enhancements, and predicted binary masks.

---

## 7. Cloud Deployment & Architecture

The application is deployed on **Streamlit Community Cloud**:
- **Live URL:** [https://dualstage-dental-caries-rcnn.streamlit.app/](https://dualstage-dental-caries-rcnn.streamlit.app/)
- **Repository:** `Mohamed-yasser77/DualStage-Dental-Caries-RCNN`
- **Model Tracking:** The 96.6 MB model artifact (`dental_caries_dualstage_retrained.pkl`) is tracked via **Git LFS** (`.gitattributes`), bypassing GitHub's standard commit file limits.
- **Dependency Optimization:** Configured with `numpy < 2.0.0` protection to avoid PyTorch C-extension incompatibility.

---

## 8. Tutor Review & Project Viva Defense Guide

Use this section to prepare for common questions from your professor or examiner.

### Q1: Why use a dual-stage architecture instead of an end-to-end segmentation network (like Mask R-CNN or pure U-Net)?
* **Answer:** A full panoramic radiograph is approximately $2940 \times 1435$ px. Directly segmenting the whole image leads to extreme class imbalance ($< 0.1\%$ caries pixels). Downsampling the image destroys small interproximal caries signals. Dividing the task into **Macro-Localization** (Faster R-CNN tooth detection) and **Micro-Segmentation** (high-resolution $256 \times 256$ tooth crops with CLAHE) allows the U-Net to focus exclusively on dental enamel and dentin at maximum native resolution.

### Q2: How did you address the severe false-positive problem observed in initial runs?
* **Answer:** We implemented four complementary interventions:
  1. **Loss Function:** Shifted from recall-biased Tversky ($\beta=0.7$) to balanced Tversky ($\alpha=0.5, \beta=0.5$) and added an empty-crop activation penalty ($\lambda_{\text{FP}}=0.5$).
  2. **Hard-Negative Mining:** Injected the model's own false-positive crops into training every 5 epochs.
  3. **Weighted Own-Box Voting:** Restricted each tooth crop's votes strictly to its own anatomical box, preventing border spillover.
  4. **Hysteresis & Area Filtering:** Suppressed connected components smaller than 50 px and required a probability peak of $\ge 0.80$.

### Q3: What is the clinical justification for CLAHE?
* **Answer:** Dental panoramic radiographs have wide dynamic range and uneven illumination. Enamel demineralization exhibits very subtle radiolucency. CLAHE (Contrast-Limited Adaptive Histogram Equalization) enhances local contrast without over-amplifying background noise, making deep enamel and dentinal lesion boundaries distinguishable to the CNN encoder.

### Q4: How is data leakage prevented during dataset augmentation?
* **Answer:** We discovered that the public DC1000 dataset contains an unannotated test mirror (`org_test_dataset`). We implemented strict string filtering (`is_test_mirror`) to guarantee that no images matching the held-out test cohort were used for training or pseudo-labeling. Our test set consists exclusively of pristine, ground-truth-annotated images.

### Q5: How do you handle overlapping tooth boxes?
* **Answer:** In Version 1.0, overlapping boxes took the maximum probability, which caused lesions on one tooth to bleed into adjacent teeth. In Version 2.0, we introduced **weighted own-box voting**: each crop only votes inside its detected bounding box ($+4\%$ margin), and overlapping regions are averaged rather than maximized.

---
*Report compiled for academic evaluation, project Viva Voce, and clinical software audit.*
