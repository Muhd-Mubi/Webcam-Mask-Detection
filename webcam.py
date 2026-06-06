#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Advanced Real-Time Face Mask Detection System  v2.0              ║
║                                                                              ║
║  Improvements over baseline:                                                 ║
║  • Deep-learning face detector (SSD ResNet-10 via OpenCV DNN)               ║
║  • Optional MediaPipe BlazeFace detector                                     ║
║  • Batched GPU inference with FP16 + TorchScript JIT                        ║
║  • Threaded frame capture (zero-lag, always-fresh frames)                    ║
║  • IOU-based face tracking + temporal prediction smoothing                   ║
║  • Live compliance stats, FPS counter, session summary                       ║
║  • MP4 recording, screenshot (S key), stats reset (R key)                   ║
║  • JSON stats export, structured logging                                     ║
║  • Full CLI via argparse  — run with --help for all options                  ║
║  • Multi-source: webcam, video file, RTSP/HTTP stream                       ║
║  • Red-border alert when any no-mask face is detected                        ║
║  • Graceful fallback chain:  MediaPipe → DNN SSD → Haar Cascade             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Install requirements (beyond baseline torch / torchvision / opencv-python):
    pip install mediapipe          # optional — fastest detector
    pip install numpy pillow       # almost certainly already present

Usage examples:
    python mask_detection_pro.py                        # webcam 0, DNN detector
    python mask_detection_pro.py --source 1             # webcam 1
    python mask_detection_pro.py --source video.mp4     # video file
    python mask_detection_pro.py --detector mediapipe   # use MediaPipe
    python mask_detection_pro.py --record --alert       # record + red alert
    python mask_detection_pro.py --no-fp16 --no-jit     # CPU-safe mode
    python mask_detection_pro.py --help
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
import urllib.request
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# 1.  STRUCTURED LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    fmt = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt,
                        datefmt="%H:%M:%S", handlers=handlers)
    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "PIL", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("MaskDetector")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CONFIGURATION DATACLASS  — every tunable knob in one place
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Model ────────────────────────────────────────────────────────────────
    model_path:  str = "mask_resnet101.pth"
    num_classes: int = 2          # 0 = Mask, 1 = No Mask
    use_fp16:    bool = True      # Half-precision on GPU (≈2× throughput)
    use_jit:     bool = True      # TorchScript trace compilation

    # ── Input source ─────────────────────────────────────────────────────────
    source: Any = 0               # int → webcam, str → file / RTSP URL

    # ── Face detection ───────────────────────────────────────────────────────
    detector_backend: str  = "dnn"   # "dnn" | "haar" | "mediapipe"
    dnn_confidence:   float = 0.60   # Minimum score for DNN detector
    face_padding:     float = 0.20   # Fractional padding around detected face
    min_face_px:      int   = 60     # Skip faces smaller than this (pixels)

    # ── Inference ────────────────────────────────────────────────────────────
    batch_size:    int = 8   # Max faces per forward pass
    smooth_window: int = 5   # Rolling frames for temporal vote smoothing

    # ── Display ──────────────────────────────────────────────────────────────
    show_fps:        bool = True
    show_stats:      bool = True
    show_confidence: bool = True
    show_landmark:   bool = False   # reserved for future face-mesh overlay
    window_name:     str  = "Face Mask Detection  ·  Q quit  ·  S screenshot  ·  R reset"

    # ── Recording ────────────────────────────────────────────────────────────
    record:      bool = False
    output_path: str  = "output.mp4"

    # ── Alerts ───────────────────────────────────────────────────────────────
    alert_no_mask: bool = False   # Red pulsing border when no-mask face found

    # ── Performance ──────────────────────────────────────────────────────────
    frame_queue_size: int = 4
    inference_skip:   int = 0    # Process every (N+1)th frame; 0 = all frames

    # ── Stats export ─────────────────────────────────────────────────────────
    export_stats: bool = False
    stats_path:   str  = "session_stats.json"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DNN FACE-DETECTOR ASSET MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

_DNN_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_DNN_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)
_DNN_PROTOTXT  = "face_deploy.prototxt"
_DNN_CAFFEMODEL = "face_ssd_resnet10.caffemodel"


def _download_dnn_assets(log: logging.Logger) -> bool:
    """Download OpenCV SSD face-detector weights if not already present."""
    pairs = [(_DNN_PROTOTXT, _DNN_PROTOTXT_URL), (_DNN_CAFFEMODEL, _DNN_MODEL_URL)]
    for path, url in pairs:
        if Path(path).exists():
            continue
        log.info(f"Downloading {path} …")
        try:
            urllib.request.urlretrieve(url, path)
            log.info(f"  ✓ saved {path}  ({Path(path).stat().st_size / 1024:.1f} KB)")
        except Exception as exc:
            log.warning(f"  ✗ download failed: {exc}")
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FACE DETECTORS  (strategy pattern — hot-swappable at runtime)
# ══════════════════════════════════════════════════════════════════════════════

FaceBox = Tuple[int, int, int, int]   # (x, y, w, h)


class BaseFaceDetector:
    """Abstract interface all detectors implement."""
    name: str = "base"

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class HaarCascadeDetector(BaseFaceDetector):
    """
    Classic Haar cascade — included as a guaranteed fallback.
    Fast but struggles with non-frontal faces, occlusion, and dark frames.
    """
    name = "haar"

    def __init__(self):
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)           # improves detection in low light
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [tuple(f) for f in faces] if len(faces) > 0 else []


# ─────────────────────────────────────────────────────────────────────────────
class DNNFaceDetector(BaseFaceDetector):
    """
    OpenCV DNN face detector — SSD with ResNet-10 backbone, Caffe model.
    Far more robust than Haar: handles profile angles, masks, poor lighting.
    No additional pip install required beyond opencv-python.
    Auto-downloads the 2.7 MB Caffe weights on first run.
    """
    name = "dnn"

    def __init__(self, confidence: float = 0.60, log: Optional[logging.Logger] = None):
        self._threshold = confidence
        self._log = log or logging.getLogger("DNNDetector")
        self._net: Optional[cv2.dnn_Net] = None
        self._load()

    def _load(self):
        if not _download_dnn_assets(self._log):
            self._log.warning("DNN assets unavailable — Haar fallback active")
            return
        try:
            self._net = cv2.dnn.readNetFromCaffe(_DNN_PROTOTXT, _DNN_CAFFEMODEL)
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._log.info("DNN face detector (SSD ResNet-10) loaded ✓")
        except Exception as exc:
            self._log.warning(f"DNN load error ({exc}) — Haar fallback active")

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        if self._net is None:
            return HaarCascadeDetector().detect(frame)

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), scalefactor=1.0,
            size=(300, 300), mean=(104.0, 177.0, 123.0),
            swapRB=False, crop=False
        )
        self._net.setInput(blob)
        detections = self._net.forward()          # shape: (1,1,N,7)

        boxes: List[FaceBox] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self._threshold:
                continue
            raw = detections[0, 0, i, 3:7] * np.array([w, h, w, h], dtype=float)
            x1, y1, x2, y2 = raw.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw > 0 and bh > 0:
                boxes.append((x1, y1, bw, bh))
        return boxes


# ─────────────────────────────────────────────────────────────────────────────
class MediaPipeFaceDetector(BaseFaceDetector):
    """
    Google MediaPipe BlazeFace detector.
    Extremely fast, ML-based, excellent for real-time on CPU.
    Requires:  pip install mediapipe
    Falls back to DNN if mediapipe is not installed.
    """
    name = "mediapipe"

    def __init__(self, confidence: float = 0.60, log: Optional[logging.Logger] = None):
        self._log = log or logging.getLogger("MediaPipeDetector")
        self._detector = None
        self._load(confidence)

    def _load(self, confidence: float):
        try:
            import mediapipe as mp  # type: ignore
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,                    # 1 = full-range (≤5 m)
                min_detection_confidence=confidence
            )
            self._log.info("MediaPipe BlazeFace detector loaded ✓")
        except ImportError:
            self._log.warning(
                "mediapipe not installed (pip install mediapipe) — DNN fallback active"
            )

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        if self._detector is None:
            return DNNFaceDetector().detect(frame)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        boxes: List[FaceBox] = []
        if results.detections:
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x  = max(0, int(bb.xmin * w))
                y  = max(0, int(bb.ymin * h))
                bw = int(bb.width  * w)
                bh = int(bb.height * h)
                if bw > 0 and bh > 0:
                    boxes.append((x, y, bw, bh))
        return boxes


def build_detector(cfg: Config, log: logging.Logger) -> BaseFaceDetector:
    """Instantiate the requested detector with graceful fallback chain."""
    backend = cfg.detector_backend.lower()
    if backend == "mediapipe":
        d = MediaPipeFaceDetector(cfg.dnn_confidence, log)
        if d._detector is not None:
            return d
        log.warning("MediaPipe unavailable — trying DNN")
        backend = "dnn"
    if backend == "dnn":
        d = DNNFaceDetector(cfg.dnn_confidence, log)
        if d._net is not None:
            return d
        log.warning("DNN unavailable — falling back to Haar")
    log.info("Using Haar Cascade detector")
    return HaarCascadeDetector()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MASK CLASSIFIER  — ResNet-101 with optional FP16 + JIT
# ══════════════════════════════════════════════════════════════════════════════

class MaskClassifier:
    """
    Wraps the pre-trained ResNet-101 with:
      • FP16 half-precision on CUDA (≈ 2× throughput)
      • TorchScript JIT trace compilation (reduced Python overhead)
      • 3-pass warm-up (eliminates first-inference spike)
      • Batched inference  (all faces in one .forward() call)
      • Softmax confidence scores (not just argmax)
    """

    LABELS       = {0: "Mask", 1: "No Mask"}
    IMGNET_MEAN  = [0.485, 0.456, 0.406]
    IMGNET_STD   = [0.229, 0.224, 0.225]
    INPUT_SIZE   = 224

    def __init__(self, cfg: Config, log: logging.Logger):
        self._cfg = cfg
        self._log = log
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16   = cfg.use_fp16 and self.device.type == "cuda"
        self._log.info(
            f"Classifier │ device={self.device} │ FP16={self.fp16} │ JIT={cfg.use_jit}"
        )
        self.model     = self._build()
        self.transform = self._build_transform()

    # ── Model construction ────────────────────────────────────────────────────
    def _build(self) -> nn.Module:
        base = models.resnet101(pretrained=False)
        base.fc = nn.Linear(base.fc.in_features, self._cfg.num_classes)

        # Load weights — handle DataParallel 'module.' prefix automatically
        raw = torch.load(self._cfg.model_path, map_location=self.device)
        clean = OrderedDict((k.replace("module.", ""), v) for k, v in raw.items())
        base.load_state_dict(clean)
        self._log.info(f"Weights loaded from '{self._cfg.model_path}'")

        base = base.to(self.device)
        if self.fp16:
            base = base.half()
            self._log.info("Model converted to FP16 ✓")

        base.eval()

        # ── TorchScript JIT trace ─────────────────────────────────────────
        if self._cfg.use_jit:
            dummy = self._dummy_tensor(1)
            try:
                with torch.no_grad():
                    traced = torch.jit.trace(base, dummy)
                traced.eval()
                self._log.info("TorchScript JIT compiled ✓")
                base = traced
            except Exception as exc:
                self._log.warning(f"JIT trace failed ({exc}) — eager mode")

        # ── Warm-up passes ────────────────────────────────────────────────
        dummy = self._dummy_tensor(1)
        with torch.no_grad():
            for _ in range(3):
                base(dummy)
        self._log.info("Warm-up (3 passes) complete ✓")
        return base

    def _dummy_tensor(self, n: int) -> torch.Tensor:
        t = torch.zeros(n, 3, self.INPUT_SIZE, self.INPUT_SIZE, device=self.device)
        return t.half() if self.fp16 else t

    # ── Preprocessing ─────────────────────────────────────────────────────────
    def _build_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.IMGNET_MEAN, self.IMGNET_STD),
        ])

    # ── Batched inference ─────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_batch(
        self, pil_images: List[Image.Image]
    ) -> List[Tuple[str, float]]:
        """
        Run a single forward pass for all images.
        Returns [(label, confidence), …] in the same order.
        """
        if not pil_images:
            return []

        batch = torch.stack([self.transform(img) for img in pil_images]).to(self.device)
        if self.fp16:
            batch = batch.half()

        logits = self.model(batch)                          # (N, C)
        probs  = torch.softmax(logits.float(), dim=1)       # always FP32
        confs, preds = torch.max(probs, dim=1)

        return [
            (self.LABELS[int(p)], float(c))
            for p, c in zip(preds, confs)
        ]


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FACE TRACKER + TEMPORAL SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

class TemporalSmoother:
    """
    Associates detected faces across frames using Intersection-over-Union
    and maintains a rolling vote history per track.

    Benefit: eliminates single-frame label flicker caused by borderline-
    confidence predictions.  A face stably classified over 5 frames is far
    more reliable than a raw per-frame score.
    """

    def __init__(self, window: int = 5, iou_threshold: float = 0.30):
        self._window    = window
        self._iou_thr   = iou_threshold
        self._tracks:   Dict[int, deque]  = {}      # id → deque[(label, conf)]
        self._boxes:    Dict[int, FaceBox] = {}      # id → last box
        self._next_id   = 0

    # ── IOU helper ────────────────────────────────────────────────────────────
    @staticmethod
    def _iou(a: FaceBox, b: FaceBox) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(ax, bx);  iy = max(ay, by)
        ex = min(ax+aw, bx+bw); ey = min(ay+ah, by+bh)
        inter = max(0, ex-ix) * max(0, ey-iy)
        union = aw*ah + bw*bh - inter
        return inter / union if union > 0 else 0.0

    # ── Update and return smoothed predictions ─────────────────────────────
    def update(
        self,
        boxes: List[FaceBox],
        raw_preds: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        used: set = set()
        smoothed: List[Tuple[str, float]] = []

        for box, pred in zip(boxes, raw_preds):
            # Associate to nearest track by IOU
            best_id, best_iou = None, self._iou_thr
            for tid, tbox in self._boxes.items():
                iou = self._iou(box, tbox)
                if iou > best_iou:
                    best_id, best_iou = tid, iou

            if best_id is None:          # new track
                best_id = self._next_id
                self._next_id += 1
                self._tracks[best_id] = deque(maxlen=self._window)

            self._tracks[best_id].append(pred)
            self._boxes[best_id] = box
            used.add(best_id)

            # Majority vote over rolling window
            history = list(self._tracks[best_id])
            vote_map: Dict[str, List[float]] = {}
            for lbl, conf in history:
                vote_map.setdefault(lbl, []).append(conf)
            winner = max(vote_map, key=lambda k: len(vote_map[k]))
            mean_conf = float(np.mean(vote_map[winner]))
            smoothed.append((winner, mean_conf))

        # Prune stale / lost tracks
        for tid in [k for k in self._boxes if k not in used]:
            self._tracks.pop(tid, None)
            self._boxes.pop(tid, None)

        return smoothed


# ══════════════════════════════════════════════════════════════════════════════
# 7.  STATISTICS TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class SessionStats:
    """Accumulates per-session metrics and computes live summaries."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._t0          = time.time()
        self.frames       = 0
        self.total_faces  = 0
        self.mask_count   = 0
        self.no_mask_count = 0
        self._fps_buf: deque = deque(maxlen=60)

    def update(self, fps: float, preds: List[Tuple[str, float]]):
        self.frames += 1
        self._fps_buf.append(fps)
        for label, _ in preds:
            self.total_faces += 1
            if label == "Mask":
                self.mask_count += 1
            else:
                self.no_mask_count += 1

    @property
    def avg_fps(self) -> float:
        return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0

    @property
    def compliance_pct(self) -> float:
        total = self.mask_count + self.no_mask_count
        return self.mask_count / total * 100.0 if total else 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    def to_dict(self) -> dict:
        return {
            "elapsed_s":           round(self.elapsed, 1),
            "total_frames":        self.frames,
            "total_faces_seen":    self.total_faces,
            "mask_detections":     self.mask_count,
            "no_mask_detections":  self.no_mask_count,
            "compliance_rate_pct": round(self.compliance_pct, 2),
            "avg_fps":             round(self.avg_fps, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  VISUALISER  — every drawing operation in one place
# ══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """
    Handles all on-screen rendering:
      • Bounding box with corner markers (cyberpunk look)
      • Colour-coded label pill with confidence
      • Semi-transparent HUD panel (FPS, stats)
      • Pulsing red-border alert for no-mask detections
    """

    _COLORS = {
        "Mask":    (50,  215,  75),     # vivid green
        "No Mask": (45,   45, 230),     # vivid red (BGR)
        "panel_bg": (15,  15,  15),
        "txt":     (235, 235, 235),
    }
    _FONT       = cv2.FONT_HERSHEY_DUPLEX
    _FONT_SM    = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE = 0.62
    _THICK      = 2

    def __init__(self, cfg: Config):
        self._cfg    = cfg
        self._alert_phase = 0.0   # used to animate the alert border

    # ── Per-face drawing ──────────────────────────────────────────────────────
    def draw_face(
        self,
        frame: np.ndarray,
        box: FaceBox,
        label: str,
        conf: float,
    ) -> None:
        x, y, w, h = box
        color = self._COLORS.get(label, (180, 180, 180))

        # Bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, self._THICK)

        # Corner accent markers (gives a "targeting reticle" look)
        cl = max(12, min(w, h) // 6)
        lw = 3
        corners = [
            [(x,   y+cl), (x,   y),   (x+cl, y  )],
            [(x+w-cl, y), (x+w, y),   (x+w,  y+cl)],
            [(x,  y+h-cl),(x,   y+h), (x+cl, y+h)],
            [(x+w-cl,y+h),(x+w, y+h),(x+w,  y+h-cl)],
        ]
        for pts in corners:
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i+1], color, lw, cv2.LINE_AA)

        # Label pill
        badge = (
            f"{label}  {conf*100:.1f}%"
            if self._cfg.show_confidence else label
        )
        (tw, th), _ = cv2.getTextSize(badge, self._FONT, self._FONT_SCALE, 1)
        pad = 5
        py1 = max(0, y - th - 2*pad)
        cv2.rectangle(frame, (x, py1), (x + tw + 2*pad, y), color, cv2.FILLED)
        cv2.putText(
            frame, badge, (x + pad, y - pad),
            self._FONT, self._FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Confidence bar under the box
        bar_w = int(w * conf)
        cv2.rectangle(frame, (x, y+h+2), (x+w, y+h+6), (60, 60, 60), cv2.FILLED)
        cv2.rectangle(frame, (x, y+h+2), (x+bar_w, y+h+6), color, cv2.FILLED)

    # ── HUD overlay ───────────────────────────────────────────────────────────
    def draw_hud(self, frame: np.ndarray, stats: SessionStats, fps: float) -> None:
        lines: List[str] = []
        if self._cfg.show_fps:
            lines.append(f"FPS   {fps:5.1f}  (avg {stats.avg_fps:.1f})")
        if self._cfg.show_stats:
            lines.append(f"Faces {stats.total_faces:,}  |  {stats.elapsed:.0f}s")
            lines.append(f"Mask {stats.mask_count}   No Mask {stats.no_mask_count}")
            lines.append(f"Compliance  {stats.compliance_pct:.1f}%")

        if not lines:
            return

        lh   = 20
        pad  = 8
        pw   = 240
        ph   = len(lines) * lh + 2 * pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8+pw, 8+ph), self._COLORS["panel_bg"], cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Thin accent line on the left of the panel
        cv2.rectangle(frame, (8, 8), (11, 8+ph), (0, 180, 120), cv2.FILLED)

        for i, line in enumerate(lines):
            cv2.putText(
                frame, line, (16, 8 + pad + (i+1)*lh - 4),
                self._FONT_SM, 0.50, self._COLORS["txt"], 1, cv2.LINE_AA
            )

    # ── Alert border ──────────────────────────────────────────────────────────
    def draw_alert(self, frame: np.ndarray, trigger: bool) -> None:
        if not self._cfg.alert_no_mask or not trigger:
            self._alert_phase = 0.0
            return
        self._alert_phase = (self._alert_phase + 0.18) % (2 * np.pi)
        alpha = 0.45 + 0.45 * np.sin(self._alert_phase)
        h, w  = frame.shape[:2]
        overlay = frame.copy()
        thickness = 10
        cv2.rectangle(overlay, (0, 0), (w-1, h-1), (0, 0, 220), thickness)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  THREADED CAPTURE  — always-fresh frames, no buffer lag
# ══════════════════════════════════════════════════════════════════════════════

class ThreadedCapture:
    """
    Reads frames in a background daemon thread and buffers them in a Queue.

    Problem with naive `cap.read()` in a loop:
        VideoCapture buffers several frames internally.  By the time you call
        `read()` the frame may already be 2–3 frames stale, causing visible lag.

    Solution:
        A background thread drains the capture continuously.  The main thread
        always pops the *most recent* frame from the queue, discarding stale ones.
    """

    def __init__(self, source: Any, maxsize: int = 4):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")
        # Shrink the internal decode buffer to 1 to minimise latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._q     = queue.Queue(maxsize=maxsize)
        self._stop  = threading.Event()
        self._t     = threading.Thread(target=self._reader, daemon=True, name="CaptureThread")
        self._t.start()

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def width(self)  -> int:   return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    @property
    def height(self) -> int:   return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    @property
    def native_fps(self) -> float:
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    # ── Background reader ─────────────────────────────────────────────────────
    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._stop.set()
                break
            # Drop oldest frame if queue full — keep freshness
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put(frame)

    # ── Public API ────────────────────────────────────────────────────────────
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._stop.is_set() and self._q.empty():
            return False, None
        try:
            return True, self._q.get(timeout=1.5)
        except queue.Empty:
            return False, None

    def release(self):
        self._stop.set()
        self._t.join(timeout=2)
        self._cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# 10.  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def pad_and_crop(
    frame: np.ndarray,
    box: FaceBox,
    padding: float = 0.20
) -> Optional[Image.Image]:
    """
    Expand the detected face box by `padding` fraction of its dimensions,
    clamp to frame boundaries, and return a PIL image for the classifier.

    The extra padding ensures the model sees forehead and chin context,
    which significantly improves mask-vs-no-mask discrimination.
    """
    x, y, w, h = box
    pw  = int(w * padding)
    ph  = int(h * padding)
    H, W = frame.shape[:2]
    x1  = max(0, x - pw)
    y1  = max(0, y - ph)
    x2  = min(W, x + w + pw)
    y2  = min(H, y + h + ph)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class MaskDetectionPipeline:
    """
    Ties every component together into a clean, single-method run loop.

    Architecture (data flow):
        ThreadedCapture → face detection → pad_and_crop
            → batch inference (MaskClassifier)
            → temporal smoothing (TemporalSmoother)
            → Visualizer.draw_*
            → cv2.imshow  [+ optional VideoWriter]
    """

    def __init__(self, cfg: Config, log: logging.Logger):
        self._cfg     = cfg
        self._log     = log
        self._stats   = SessionStats()
        self._vis     = Visualizer(cfg)

        # Initialise heavy components
        self._log.info("Loading mask classifier …")
        self._clf     = MaskClassifier(cfg, log)

        self._log.info("Initialising face detector …")
        self._det     = build_detector(cfg, log)

        self._smoother = TemporalSmoother(window=cfg.smooth_window)
        self._writer:  Optional[cv2.VideoWriter] = None
        self._frame_n  = 0

    # ── VideoWriter setup ─────────────────────────────────────────────────────
    def _make_writer(self, cap: ThreadedCapture) -> Optional[cv2.VideoWriter]:
        if not self._cfg.record:
            return None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(
            self._cfg.output_path, fourcc, cap.native_fps,
            (cap.width, cap.height)
        )

    # ── Core run loop ─────────────────────────────────────────────────────────
    def run(self) -> None:
        cap = ThreadedCapture(self._cfg.source, self._cfg.frame_queue_size)
        self._log.info(
            f"Stream  {cap.width}×{cap.height}  @{cap.native_fps:.1f} fps  "
            f"source={self._cfg.source!r}"
        )
        self._writer = self._make_writer(cap)
        if self._writer:
            self._log.info(f"Recording → {self._cfg.output_path}")

        t_prev = time.perf_counter()
        fps    = 0.0

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._log.warning("Capture ended or error — stopping.")
                    break

                self._frame_n += 1

                # ── Optional frame-skip (saves GPU on slow machines) ─────────
                skip = self._cfg.inference_skip
                if skip > 0 and self._frame_n % (skip + 1) != 0:
                    cv2.imshow(self._cfg.window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                # ── Face detection ───────────────────────────────────────────
                raw_boxes = self._det.detect(frame)
                # Filter faces below minimum size
                boxes = [
                    b for b in raw_boxes
                    if b[2] >= self._cfg.min_face_px and b[3] >= self._cfg.min_face_px
                ]

                # ── Crop → batch classify ────────────────────────────────────
                raw_preds: List[Tuple[str, float]] = []
                if boxes:
                    crops = [pad_and_crop(frame, b, self._cfg.face_padding) for b in boxes]
                    valid_pairs = [(b, c) for b, c in zip(boxes, crops) if c is not None]

                    if valid_pairs:
                        boxes_v, crops_v = zip(*valid_pairs)
                        boxes = list(boxes_v)
                        all_crops = list(crops_v)
                        # Process in mini-batches
                        for i in range(0, len(all_crops), self._cfg.batch_size):
                            chunk = all_crops[i : i + self._cfg.batch_size]
                            raw_preds.extend(self._clf.predict_batch(chunk))

                # ── Temporal smoothing ───────────────────────────────────────
                smoothed = self._smoother.update(boxes, raw_preds)

                # ── FPS ──────────────────────────────────────────────────────
                now   = time.perf_counter()
                fps   = 1.0 / max(now - t_prev, 1e-9)
                t_prev = now

                # ── Stats ────────────────────────────────────────────────────
                self._stats.update(fps, smoothed)

                # ── Render ───────────────────────────────────────────────────
                no_mask = any(lbl == "No Mask" for lbl, _ in smoothed)

                for box, (label, conf) in zip(boxes, smoothed):
                    self._vis.draw_face(frame, box, label, conf)

                self._vis.draw_hud(frame, self._stats, fps)
                self._vis.draw_alert(frame, no_mask)

                # ── Record ───────────────────────────────────────────────────
                if self._writer:
                    self._writer.write(frame)

                cv2.imshow(self._cfg.window_name, frame)

                # ── Keyboard shortcuts ────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._log.info("Quit by user (Q)")
                    break
                elif key == ord("s"):
                    fname = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(fname, frame)
                    self._log.info(f"Screenshot saved → {fname}")
                elif key == ord("r"):
                    self._stats.reset()
                    self._log.info("Session stats reset")
                elif key == ord("f"):
                    self._cfg.show_fps = not self._cfg.show_fps

        finally:
            self._cleanup(cap)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def _cleanup(self, cap: ThreadedCapture) -> None:
        cap.release()
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()

        # Print session summary
        summary = self._stats.to_dict()
        bar = "═" * 55
        self._log.info(bar)
        self._log.info("  SESSION SUMMARY")
        self._log.info(bar)
        for k, v in summary.items():
            self._log.info(f"    {k:<30} {v}")
        self._log.info(bar)

        if self._cfg.export_stats:
            with open(self._cfg.stats_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
            self._log.info(f"Stats exported → {self._cfg.stats_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 12.  CLI  — every Config field is exposed as a flag
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model
    p.add_argument("--model",      default="mask_resnet101.pth", metavar="PATH",
                   help="Path to ResNet-101 .pth weights file")
    p.add_argument("--classes",    default=2, type=int,
                   help="Number of output classes (default 2: Mask / No Mask)")
    p.add_argument("--no-fp16",    action="store_true",
                   help="Disable FP16 half-precision (use if CUDA is absent)")
    p.add_argument("--no-jit",     action="store_true",
                   help="Disable TorchScript JIT compilation")

    # Source
    p.add_argument("--source",     default="0", metavar="SRC",
                   help="Webcam index, video file path, or RTSP/HTTP URL (default 0)")

    # Detection
    p.add_argument("--detector",   default="dnn", choices=["dnn", "haar", "mediapipe"],
                   help="Face detector backend (default: dnn)")
    p.add_argument("--dnn-conf",   default=0.60, type=float, metavar="THRESH",
                   help="DNN detector confidence threshold (default 0.60)")
    p.add_argument("--padding",    default=0.20, type=float,
                   help="Fractional padding around detected face (default 0.20)")
    p.add_argument("--min-face",   default=60, type=int, metavar="PX",
                   help="Ignore faces smaller than this many pixels (default 60)")

    # Inference
    p.add_argument("--batch",      default=8, type=int,
                   help="Max faces per GPU forward pass (default 8)")
    p.add_argument("--smooth",     default=5, type=int,
                   help="Temporal smoothing window in frames (default 5)")
    p.add_argument("--skip",       default=0, type=int,
                   help="Run inference every (N+1)th frame; 0 = every frame")

    # Display
    p.add_argument("--no-fps",     action="store_true",  help="Hide FPS counter")
    p.add_argument("--no-stats",   action="store_true",  help="Hide statistics panel")
    p.add_argument("--no-conf",    action="store_true",  help="Hide confidence % on labels")

    # Recording
    p.add_argument("--record",     action="store_true",  help="Record output to MP4")
    p.add_argument("--output",     default="output.mp4", help="Recording path (default output.mp4)")

    # Alerts
    p.add_argument("--alert",      action="store_true",
                   help="Pulsing red border when any no-mask face is detected")

    # Stats
    p.add_argument("--export-stats", action="store_true",
                   help="Export session stats to JSON at end")
    p.add_argument("--stats-path", default="session_stats.json",
                   help="Path for exported stats (default session_stats.json)")

    # Logging
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default INFO)")
    p.add_argument("--log-file",   default=None, metavar="PATH",
                   help="Also write logs to this file")

    return p


def main() -> None:
    args = _build_parser().parse_args()
    log  = setup_logging(args.log_level, args.log_file)

    # Coerce source to int for webcam indices
    src = args.source
    if src.isdigit():
        src = int(src)

    cfg = Config(
        model_path        = args.model,
        num_classes       = args.classes,
        use_fp16          = not args.no_fp16,
        use_jit           = not args.no_jit,
        source            = src,
        detector_backend  = args.detector,
        dnn_confidence    = args.dnn_conf,
        face_padding      = args.padding,
        min_face_px       = args.min_face,
        batch_size        = args.batch,
        smooth_window     = args.smooth,
        inference_skip    = args.skip,
        show_fps          = not args.no_fps,
        show_stats        = not args.no_stats,
        show_confidence   = not args.no_conf,
        record            = args.record,
        output_path       = args.output,
        alert_no_mask     = args.alert,
        export_stats      = args.export_stats,
        stats_path        = args.stats_path,
    )

    log.info("╔═══════════════════════════════════════╗")
    log.info("║   Face Mask Detection System  v2.0    ║")
    log.info("╚═══════════════════════════════════════╝")
    log.info(f"Detector : {cfg.detector_backend.upper()}   |   "
             f"FP16: {cfg.use_fp16}   |   JIT: {cfg.use_jit}   |   "
             f"Smooth window: {cfg.smooth_window}")

    try:
        pipeline = MaskDetectionPipeline(cfg, log)
        pipeline.run()
    except RuntimeError as exc:
        log.error(f"Fatal: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl-C)")


if __name__ == "__main__":
    main()