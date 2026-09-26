"""Shared detect-stage definitions and load_detector(), which returns the
text-map detector (ml/textseg/detector.py) for a textseg ONNX export. The
letterbox mirrors the extension's letterbox.ts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The model input is letterboxed to this square size.
INPUT_SIZE = 1280
# Neutral grey letterbox padding.
PAD_COLOR = (114, 114, 114)

# Detection runs at this floor and the user's threshold filters the result,
# so recall is measured at the floor.
DETECT_FLOOR = 0.05
# The extension's default threshold (settings.ts detectThreshold).
DEFAULT_THRESHOLD = 0.6
DEFAULT_IOU = 0.5


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float


def area(box: tuple[float, float, float, float]) -> float:
    return box[2] * box[3]


def intersection(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    overlap = intersection(a, b)
    if overlap <= 0:
        return 0.0
    union = area(a) + area(b) - overlap
    return overlap / union if union > 0 else 0.0


def containment(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Fraction of the smaller box lying inside the larger."""
    smaller = min(area(a), area(b))
    return intersection(a, b) / smaller if smaller > 0 else 0.0


def encloses(
    outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]
) -> float:
    """Directional: fraction of `inner` inside `outer`."""
    size = area(inner)
    return intersection(outer, inner) / size if size > 0 else 0.0


def load_detector(model: Path):
    """The text-map detector for a textseg ONNX export."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "textseg"))
    from detector import TextMapDetector

    return TextMapDetector(model)
