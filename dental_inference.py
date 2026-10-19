"""
dental_inference.py  --  Dual-stage dental caries segmentation (inference + model bundle I/O).

Stage 1: Faster R-CNN (MobileNetV3-Large FPN) localises individual teeth in a panoramic radiograph.
Stage 2: U-Net (EfficientNet-B0 encoder) segments carious pixels inside each tooth crop.

Usage (outside Kaggle):
    from dental_inference import DualStagePredictor
    pred = DualStagePredictor.from_pickle("dental_caries_dualstage.pkl")
    out = pred.predict("some_panoramic.png")      # dict: mask, prob, boxes, scores
    cv2.imwrite("mask.png", out["mask"] * 255)

Requires: torch, torchvision, segmentation-models-pytorch, opencv-python, numpy.
"""
import os
import pickle
import warnings

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- cropping utilities
def expand_box(box, W, H, pad=0.05):
    """Expand an (x1,y1,x2,y2) box by `pad` (fraction of its size) and clip to the image. Returns ints."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(np.floor(x1 - pad * w)))
    y1 = max(0, int(np.floor(y1 - pad * h)))
    x2 = min(W, int(np.ceil(x2 + pad * w)))
    y2 = min(H, int(np.ceil(y2 + pad * h)))
    return x1, y1, x2, y2


def crop_tooth(gray, box, size=256, pad=0.05):
    """Crop one tooth from a uint8 grayscale panoramic image, apply CLAHE, resize to size x size.
    Returns (crop_uint8 [size,size], padded_box) or (None, None) if the box is degenerate."""
    H, W = gray.shape[:2]
    x1, y1, x2, y2 = expand_box(box, W, H, pad)
    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return None, None
    c = gray[y1:y2, x1:x2]
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(c)
    interp = cv2.INTER_AREA if min(c.shape[:2]) > size else cv2.INTER_LINEAR
    c = cv2.resize(c, (size, size), interpolation=interp)
    return c, (x1, y1, x2, y2)


def crop_mask(mask01, padded_box, size=256):
    """Crop + resize a binary {0,1} uint8 mask with the same padded box used for the image crop."""
    x1, y1, x2, y2 = padded_box
    m = cv2.resize(mask01[y1:y2, x1:x2] * 255, (size, size), interpolation=cv2.INTER_AREA)
    return (m > 127).astype(np.uint8)


def prep(xb):
    """(B,1,S,S) float in [0,1] -> (B,3,S,S) ImageNet-normalised (gray replicated to 3 channels)."""
    mean = torch.tensor(IMAGENET_MEAN, device=xb.device, dtype=xb.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=xb.device, dtype=xb.dtype).view(1, 3, 1, 1)
    return (xb.repeat(1, 3, 1, 1) - mean) / std


def postprocess_prob(prob, lo=0.5, hi=None, min_area=0):
    """Probability map -> binary mask with false-positive suppression.
    * hysteresis: pixels > `lo` form candidate blobs; a blob survives only if its peak probability >= `hi`
      (hi=None/<=lo disables the peak requirement),
    * blobs smaller than `min_area` pixels (original image resolution) are dropped."""
    low = (prob > lo).astype(np.uint8)
    if not low.any():
        return low
    n, lab, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    if n <= 1:
        return low
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    if hi is not None and hi > lo:
        ys, xs = np.nonzero(low)
        peak = np.zeros(n, np.float32)
        np.maximum.at(peak, lab[ys, xs], prob[ys, xs])
        keep &= peak >= hi
    keep[0] = False
    return keep[lab].astype(np.uint8)


# ----------------------------------------------------------------------------- model builders
def build_unet(encoder="efficientnet-b0", pretrained=True, verbose=True):
    """U-Net with an EfficientNet-B0 encoder (via segmentation_models_pytorch).
    Tries several equivalent encoder names because they differ between smp versions.
    Returns (model, encoder_name_used, got_pretrained_weights)."""
    import segmentation_models_pytorch as smp

    names = [encoder] + [n for n in ("efficientnet-b0", "timm-efficientnet-b0", "tu-efficientnet_b0") if n != encoder]
    weight_options = (["imagenet"] if pretrained else []) + [None]
    last_err = None
    for w in weight_options:
        for name in names:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = smp.Unet(encoder_name=name, encoder_weights=w, in_channels=3, classes=1)
                if verbose:
                    print(f"[build_unet] encoder={name} weights={w}")
                    if pretrained and w is None:
                        print("  !! WARNING: ImageNet weights could NOT be loaded (is Internet ON?). "
                              "Training from scratch -> expect worse results.")
                return m, name, (w == "imagenet")
            except Exception as e:  # noqa
                last_err = e
    raise RuntimeError(f"Could not build U-Net: {last_err}")


def build_detector(pretrained=True, min_size=800, max_size=1700, verbose=True):
    """Faster R-CNN with MobileNetV3-Large FPN backbone, 2 classes (background + tooth).
    Returns (model, got_pretrained_backbone)."""
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

    kw = dict(num_classes=2, min_size=min_size, max_size=max_size, box_detections_per_img=64)
    if pretrained:
        try:
            from torchvision.models import MobileNet_V3_Large_Weights
            m = fasterrcnn_mobilenet_v3_large_fpn(
                weights=None, weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1, **kw)
            if verbose:
                print("[build_detector] MobileNetV3-Large FPN, ImageNet backbone weights loaded")
            return m, True
        except Exception as e:  # noqa
            print(f"  !! WARNING: could not load ImageNet backbone weights ({type(e).__name__}). "
                  "Is Internet ON? Training the detector from scratch.")
    m = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, **kw)
    return m, False


# ----------------------------------------------------------------------------- predictor
class DualStagePredictor:
    """Detector -> per-tooth crops -> U-Net -> stitched panoramic mask."""

    def __init__(self, unet, detector, size=256, pad=0.05, det_thr=0.5, unet_thr=0.5,
                 unet_encoder="efficientnet-b0", det_min_size=800, det_max_size=1700, device=None,
                 unet_hi=None, min_area=0, own_box_margin=0.04, merge="weighted"):
        self.device = torch.device(device) if device else next(unet.parameters()).device
        self.unet = unet.to(self.device).eval()
        self.det = detector.to(self.device).eval()
        self.size, self.pad = size, pad
        self.det_thr, self.unet_thr = det_thr, unet_thr
        # false-positive suppression: unet_thr = low (grow) threshold, unet_hi = required blob peak,
        # min_area = smallest blob kept (px), merge = 'weighted' (own-box vote, averaged) or 'max' (legacy)
        self.unet_hi, self.min_area = unet_hi, min_area
        self.own_box_margin, self.merge = own_box_margin, merge
        self.unet_encoder = unet_encoder
        self.det_min_size, self.det_max_size = det_min_size, det_max_size

    # -- helpers
    @staticmethod
    def _load_gray(image):
        if isinstance(image, (str, os.PathLike)):
            g = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise IOError(f"cannot read image: {image}")
            return g
        a = np.asarray(image)
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        return a

    @property
    def _amp(self):
        return self.device.type == "cuda"

    # -- stage 1
    @torch.no_grad()
    def detect(self, gray, thr=None):
        thr = self.det_thr if thr is None else thr
        x = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1).to(self.device)
        o = self.det([x])[0]
        k = o["scores"] >= thr
        return o["boxes"][k].float().cpu().numpy(), o["scores"][k].float().cpu().numpy()

    # -- stage 2
    @torch.no_grad()
    def segment_crops(self, crops_u8, bs=32, tta_flip=True):
        """crops_u8: (N,S,S) uint8 -> (N,S,S) float32 probabilities.
        tta_flip=True averages each crop's prediction with its horizontally-flipped version
        (test-time augmentation): ~2x U-Net compute, no retraining needed, usually a small free Dice gain."""
        out = []
        for i in range(0, len(crops_u8), bs):
            xb = torch.from_numpy(np.ascontiguousarray(crops_u8[i:i + bs])).float().div(255).unsqueeze(1).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self._amp):
                lg = self.unet(prep(xb)).float()
                if tta_flip:
                    lg_f = torch.flip(self.unet(prep(torch.flip(xb, dims=[-1]))).float(), dims=[-1])
                    lg = (lg + lg_f) / 2
            out.append(torch.sigmoid(lg).squeeze(1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.size, self.size), np.float32)

    # -- full pipeline
    def _own_region(self, box, pbox, W, H):
        """0/1 weight over the padded crop window: 1 only inside this tooth's own box (+ margin)."""
        px1, py1, px2, py2 = pbox
        ox1, oy1, ox2, oy2 = expand_box(box, W, H, self.own_box_margin)
        w = np.zeros((py2 - py1, px2 - px1), np.float32)
        w[max(oy1, py1) - py1:min(oy2, py2) - py1, max(ox1, px1) - px1:min(ox2, px2) - px1] = 1.0
        return w

    @torch.no_grad()
    def predict_prob(self, image, boxes=None):
        """Stages 1+2 without thresholding. Returns (prob float32 HxW, boxes (N,4), scores (N,))."""
        gray = self._load_gray(image)
        H, W = gray.shape
        if boxes is None:
            boxes, scores = self.detect(gray)
        else:
            boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
            scores = np.ones(len(boxes), np.float32)
        crops, pboxes, oboxes = [], [], []
        for b in boxes:
            c, pb = crop_tooth(gray, b, self.size, self.pad)
            if c is not None:
                crops.append(c)
                pboxes.append(pb)
                oboxes.append(b)
        prob = np.zeros((H, W), np.float32)
        if crops:
            probs = self.segment_crops(np.stack(crops))
            if self.merge == "max":                      # legacy: any crop firing wins (padding context included)
                for p, (x1, y1, x2, y2) in zip(probs, pboxes):
                    pr = cv2.resize(p, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                    prob[y1:y2, x1:x2] = np.maximum(prob[y1:y2, x1:x2], pr)
            else:                                        # each crop votes only inside its own tooth box; votes averaged
                acc = np.zeros((H, W), np.float32)
                wsum = np.zeros((H, W), np.float32)
                for p, pb, ob in zip(probs, pboxes, oboxes):
                    x1, y1, x2, y2 = pb
                    pr = cv2.resize(p, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                    w = self._own_region(ob, pb, W, H)
                    acc[y1:y2, x1:x2] += pr * w
                    wsum[y1:y2, x1:x2] += w
                np.divide(acc, wsum, out=prob, where=wsum > 0)
        return prob, np.asarray(boxes), np.asarray(scores)

    @torch.no_grad()
    def predict(self, image, boxes=None):
        """image: path or ndarray. boxes: optional (N,4) xyxy tooth boxes (skips the detector, e.g. ground truth).
        Returns dict(mask uint8 {0,1} HxW, prob float32 HxW, boxes (N,4), scores (N,))."""
        prob, boxes, scores = self.predict_prob(image, boxes)
        mask = postprocess_prob(prob, self.unet_thr, self.unet_hi, self.min_area)
        return dict(mask=mask, prob=prob, boxes=boxes, scores=scores)

    # -- (de)serialisation: one pickle file holding everything needed to rebuild the pipeline
    def to_bundle(self, extra=None):
        cpu = lambda sd: {k: v.detach().cpu() for k, v in sd.items()}  # noqa: E731

        # The predictor normally receives the underlying detector, but keep export
        # robust if a DataParallel/wrapper object is passed in.
        det_model = self.det
        if hasattr(det_model, "module"):
            det_model = det_model.module
        if hasattr(det_model, "detector"):
            det_model = det_model.detector

        unet_model = self.unet
        if hasattr(unet_model, "module"):
            unet_model = unet_model.module

        return dict(
            format_version=2,
            unet=dict(encoder=self.unet_encoder, state_dict=cpu(unet_model.state_dict())),
            detector=dict(
                state_dict=cpu(det_model.state_dict()),
                min_size=self.det_min_size,
                max_size=self.det_max_size,
            ),
            settings=dict(
                size=self.size, pad=self.pad,
                det_thr=self.det_thr, unet_thr=self.unet_thr,
                unet_hi=self.unet_hi, min_area=self.min_area,
                own_box_margin=self.own_box_margin, merge=self.merge,
                mean=IMAGENET_MEAN, std=IMAGENET_STD,
            ),
            extra=extra or {},
        )

    def save_pickle(self, path, extra=None):
        with open(path, "wb") as f:
            pickle.dump(self.to_bundle(extra), f, protocol=4)
        return path

    @classmethod
    def from_bundle(cls, bundle, device=None):
        device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet, enc, _ = build_unet(bundle["unet"]["encoder"], pretrained=False, verbose=False)
        unet.load_state_dict(bundle["unet"]["state_dict"])
        d = bundle["detector"]
        det, _ = build_detector(pretrained=False, min_size=d["min_size"], max_size=d["max_size"], verbose=False)
        det.load_state_dict(d["state_dict"])
        s = bundle["settings"]
        return cls(unet, det, size=s["size"], pad=s["pad"], det_thr=s["det_thr"], unet_thr=s["unet_thr"],
                   unet_encoder=enc, det_min_size=d["min_size"], det_max_size=d["max_size"], device=device,
                   unet_hi=s.get("unet_hi"), min_area=s.get("min_area", 0),
                   own_box_margin=s.get("own_box_margin", 0.04), merge=s.get("merge", "max"))  # v1 bundles -> legacy behaviour

    @classmethod
    def from_pickle(cls, path, device=None):
        with open(path, "rb") as f:
            return cls.from_bundle(pickle.load(f), device)