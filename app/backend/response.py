"""
Processes raw inference output into the application's detection format.

This module is backend-agnostic: it consumes raw model outputs (e.g., YOLO-format
tensors from inference servers) and produces structured detections suitable for
the PPE compliance pipeline. The output is used to draw bounding boxes around
detected objects in the video feed. It handles:

- Parsing raw tensors into Detection objects (with NMS, confidence filtering)
- Converting bounding boxes to pixel coordinates
- Filtering classes and formatting for display and tracking

Using a separate module keeps inference backend details (OVMS, Triton, etc.)
decoupled from the rest of the application. If the inference backend changes,
only this module and the runtime need to be updated.
"""

import os

import cv2
import numpy as np
from collections import defaultdict
from pydantic import BaseModel

from logger import get_logger

log = get_logger(__name__)

# YOLO_CLASS_SIGMOID: true | false | auto (default). "auto" applies sigmoid to class
# channels when values look like logits (outside [0,1] or negative). Some OpenVINO
# exports emit probabilities already; forcing true on those will break scores.
#
# Pre-NMS class score floor and cv2.dnn.NMSBoxes score threshold are fixed below.
_YOLO_MIN_CLASS_SCORE_BEFORE_NMS = 0.25
_YOLO_NMS_SCORE_THRESHOLD = 0.20


class Detection(BaseModel):
    """A single detection from the inference model."""

    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x, y, w, h] in model coordinates
    scale: float


def _raw_prediction_tensor(outputs) -> np.ndarray:
    """First output tensor from OVMS / KServe (dict, list, or ndarray)."""
    if isinstance(outputs, dict):
        arr = outputs.get("output0")
        if arr is None:
            arr = next(iter(outputs.values()))
    elif isinstance(outputs, (list, tuple)):
        arr = outputs[0]
    else:
        arr = outputs
    return np.asarray(arr)


def _sigmoid(class_scores: np.ndarray) -> np.ndarray:
    x = np.clip(class_scores.astype(np.float64), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def _apply_class_sigmoid(class_scores: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (scores, applied) per YOLO_CLASS_SIGMOID env."""
    mode = (os.environ.get("YOLO_CLASS_SIGMOID") or "auto").strip().lower()
    if mode in ("1", "true", "yes"):
        return _sigmoid(class_scores), True
    if mode in ("0", "false", "no"):
        return class_scores, False
    # auto: logits often outside [0, 1] or negative
    if class_scores.size == 0:
        return class_scores, False
    cmax = float(np.max(class_scores))
    cmin = float(np.min(class_scores))
    if cmax > 1.0 + 1e-6 or cmin < -1e-6:
        return _sigmoid(class_scores), True
    return class_scores, False


def _predictions_matrix(raw: np.ndarray, nc: int) -> np.ndarray:
    """
    Return float32 array of shape (num_predictions, 4 + nc) — one row per anchor.

    Ultralytics detect export is usually (1, 4+nc, N) or (1, N, 4+nc); we normalize
    to (N, 4+nc).
    """
    x = np.asarray(raw)
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(
            f"Expected YOLO output to reduce to 2D (predictions × features); got shape {raw.shape!r}"
        )

    feat = 4 + nc

    if x.shape[0] == feat:
        data = x.T
    elif x.shape[1] == feat:
        data = x
    elif x.shape[0] > x.shape[1] and x.shape[1] == feat:
        data = x
    elif x.shape[1] > x.shape[0] and x.shape[0] == feat:
        data = x.T
    else:
        log.warning(
            "YOLO layout ambiguous shape=%s expected feat_dim=%d; "
            "using short-axis-as-features heuristic — check OVMS tensor layout vs model export",
            x.shape,
            feat,
        )
        data = x.T if x.shape[0] < x.shape[1] else x

    if data.shape[1] != feat:
        raise ValueError(
            f"YOLO feature dimension mismatch: got {data.shape[1]} columns, "
            f"expected {feat} (4 box + {nc} classes). Check export nc vs app config classes."
        )

    return data.astype(np.float32)


def postprocess_image(
    outputs, scale: float, classes: dict[int, str]
) -> list[Detection]:
    """
    Parse raw inference tensor into Detection objects.

    Expects YOLO-style output format. Applies a fixed pre-NMS class score floor
    and NMS score threshold for ``cv2.dnn.NMSBoxes`` (see module constants
    ``_YOLO_MIN_CLASS_SCORE_BEFORE_NMS`` and ``_YOLO_NMS_SCORE_THRESHOLD``).

    Args:
        outputs: Raw model output (numpy array, list/tuple of arrays, or dict).
        scale: Scale factor from preprocessing (model coords → original image).
        classes: Mapping of class_id to class_name.

    Returns:
        List of Detection objects.
    """
    nc = len(classes)
    raw = _raw_prediction_tensor(outputs)
    data = _predictions_matrix(raw, nc)

    class_scores = data[:, 4 : 4 + nc]
    class_scores, _ = _apply_class_sigmoid(class_scores)

    max_scores = class_scores.max(axis=1)
    max_class_ids = class_scores.argmax(axis=1)

    min_class_conf = _YOLO_MIN_CLASS_SCORE_BEFORE_NMS
    nms_score_thr = _YOLO_NMS_SCORE_THRESHOLD

    mask = max_scores >= min_class_conf
    data = data[mask]
    scores = max_scores[mask]
    class_ids = max_class_ids[mask]

    cx, cy, w, h = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    boxes = np.column_stack([cx - 0.5 * w, cy - 0.5 * h, w, h])

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, nms_score_thr, 0.45, 0.5)

    detections: list[Detection] = []
    for idx in result_boxes:
        detections.append(
            Detection(
                class_id=int(class_ids[idx]),
                class_name=classes[int(class_ids[idx])],
                confidence=float(scores[idx]),
                bbox=boxes[idx].tolist(),
                scale=scale,
            )
        )

    return detections


def process_detections(
    runtime_detections: list[Detection],
    include_in_counts_by_class_id: dict[int, bool] | None = None,
    trackable_by_class_id: dict[int, bool] | None = None,
) -> tuple[list[dict], defaultdict]:
    """Convert Detection objects into app format for display and tracking.

    Args:
        runtime_detections: Detections from postprocess_image or runtime.
        include_in_counts_by_class_id: Per model_class_index flag from detection_classes.
            When False, the detection is omitted (no boxes, counts, or tracker input).
            None or missing keys default to True.
        trackable_by_class_id: Unused, kept for call-site compatibility.

    Returns:
        Tuple of:
        - detections: List of dicts with bbox (x1,y1,x2,y2), confidence,
          class_id, class_name.
        - counts: defaultdict of class_name -> count.
    """
    detections = []
    counts = defaultdict(int)

    for d in runtime_detections:
        if include_in_counts_by_class_id is not None and not (
            include_in_counts_by_class_id.get(d.class_id, True)
        ):
            continue
        counts[d.class_name] += 1
        x, y, w, h = d.bbox
        x1 = round(x * d.scale)
        y1 = round(y * d.scale)
        x2 = round((x + w) * d.scale)
        y2 = round((y + h) * d.scale)
        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "confidence": d.confidence,
                "class_id": d.class_id,
                "class_name": d.class_name,
            }
        )

    return detections, counts
