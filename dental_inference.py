"""
dental_inference.py  --  Dual-stage dental caries segmentation (inference + model bundle I/O).

Stage 1: Faster R-CNN (MobileNetV3-Large FPN) localises individual teeth in a panoramic radiograph.
Stage 2: U-Net (EfficientNet-B0 encoder) segments carious pixels inside each tooth crop.

Usage:
    from dental_inference import DualStagePredictor
    pred = DualStagePredictor.from_pickle("dental_caries_dualstage (2).pkl")
    out = pred.predict("some_panoramic.png")      # dict: mask, prob, boxes, scores
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


# ----------------------------------------------------------------------------- model builders
def build_unet(encoder="efficientnet-b0", pretrained=True, verbose=True):
    """U-Net with an EfficientNet-B0 encoder (via segmentation_models_pytorch)."""
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
                return m, name, (w == "imagenet")
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Could not build U-Net: {last_err}")


def build_detector(pretrained=True, min_size=800, max_size=1700, verbose=True):
    """Faster R-CNN with MobileNetV3-Large FPN backbone, 2 classes (background + tooth)."""
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
        except Exception as e:
            print(f"  !! WARNING: could not load ImageNet backbone weights ({type(e).__name__}).")
    m = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, **kw)
    return m, False


# ----------------------------------------------------------------------------- predictor
class DualStagePredictor:
    """Detector -> per-tooth crops -> U-Net -> stitched panoramic mask."""

    def __init__(self, unet, detector, size=256, pad=0.05, det_thr=0.5, unet_thr=0.5,
                 unet_encoder="efficientnet-b0", det_min_size=800, det_max_size=1700, device=None):
        self.device = torch.device(device) if device else next(unet.parameters()).device
        self.unet = unet.to(self.device).eval()
        self.det = detector.to(self.device).eval()
        self.size, self.pad = size, pad
        self.det_thr, self.unet_thr = det_thr, unet_thr
        self.unet_encoder = unet_encoder
        self.det_min_size, self.det_max_size = det_min_size, det_max_size

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
        # FP16 causes numerical instability and NaNs with EfficientNet-B0 Swish/BatchNorm layers
        return False

    @torch.no_grad()
    def detect(self, gray, thr=None):
        thr = self.det_thr if thr is None else thr
        x = torch.from_numpy(gray).float().div(255).unsqueeze(0).repeat(3, 1, 1).to(self.device)
        o = self.det([x])[0]
        k = o["scores"] >= thr
        return o["boxes"][k].float().cpu().numpy(), o["scores"][k].float().cpu().numpy()

    @torch.no_grad()
    def segment_crops(self, crops_u8, bs=32, tta_flip=True):
        out = []
        for i in range(0, len(crops_u8), bs):
            xb = torch.from_numpy(np.ascontiguousarray(crops_u8[i:i + bs])).float().div(255).unsqueeze(1).to(self.device)
            lg = self.unet(prep(xb)).float()
            if tta_flip:
                lg_f = torch.flip(self.unet(prep(torch.flip(xb, dims=[-1]))).float(), dims=[-1])
                lg = (lg + lg_f) / 2
            out.append(torch.sigmoid(lg).squeeze(1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.size, self.size), np.float32)

    @torch.no_grad()
    def predict(self, image, boxes=None):
        gray = self._load_gray(image)
        H, W = gray.shape
        if boxes is None:
            boxes, scores = self.detect(gray)
        else:
            boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
            scores = np.ones(len(boxes), np.float32)
        crops, pboxes = [], []
        for b in boxes:
            c, pb = crop_tooth(gray, b, self.size, self.pad)
            if c is not None:
                crops.append(c)
                pboxes.append(pb)
        prob = np.zeros((H, W), np.float32)
        if crops:
            probs = self.segment_crops(np.stack(crops))
            for p, (x1, y1, x2, y2) in zip(probs, pboxes):
                pr = cv2.resize(p, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                prob[y1:y2, x1:x2] = np.maximum(prob[y1:y2, x1:x2], pr)
        return dict(mask=(prob > self.unet_thr).astype(np.uint8), prob=prob,
                    boxes=np.asarray(boxes), scores=np.asarray(scores))

    def to_bundle(self, extra=None):
        cpu = lambda sd: {k: v.detach().cpu() for k, v in sd.items()}
        det_model = self.det.module if hasattr(self.det, "module") else self.det
        unet_model = self.unet.module if hasattr(self.unet, "module") else self.unet

        return dict(
            format_version=1,
            unet=dict(encoder=self.unet_encoder, state_dict=cpu(unet_model.state_dict())),
            detector=dict(
                state_dict=cpu(det_model.state_dict()),
                min_size=self.det_min_size,
                max_size=self.det_max_size,
            ),
            settings=dict(
                size=self.size, pad=self.pad,
                det_thr=self.det_thr, unet_thr=self.unet_thr,
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
                   unet_encoder=enc, det_min_size=d["min_size"], det_max_size=d["max_size"], device=device)

    @classmethod
    def from_pickle(cls, path, device=None):
        with open(path, "rb") as f:
            return cls.from_bundle(pickle.load(f), device)
