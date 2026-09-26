"""Training samples from build_cache.py's output: a crop, its target map, its loss mask.

Target: each text box painted shrunk, DBNet's way (every side moves in by
D = A * (1 - r^2) / L, r = SHRINK_RATIO), so touching balloons stay separate
blobs; detector.py grows them back. A box too small to survive the shrink is
ignored, not taught as background.

Mask: 1 where the loss counts. Ignore boxes and ignore-everywhere negative
pages are 0. The grey letterbox padding counts as background, since inference
sees it; untrained, it produced false boxes along the page edge.

Augmentation: scale jitter, random crop, contrast and brightness, greyscale and
inversion (white-on-dark lettering is the same shapes). No horizontal flip:
mirrored kana are not lettering.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from model import OUTPUT_STRIDE

INPUT_SIZE = 1280       # inference letterbox, as in ml/eval/detect.py
PAD_VALUE = 114
SHRINK_RATIO = 0.4
MIN_SHRUNK_SIDE = 2.0   # in map pixels (stride 4); below this a box is ignore


def shrink_offset(w: float, h: float, ratio: float = SHRINK_RATIO) -> float:
    return (w * h) * (1 - ratio**2) / max(1e-6, 2 * (w + h))


def paint_targets(
    size: tuple[int, int],
    text: list[list[float]],
    ignore: list[list[float]],
    background: list[list[float]] | None,
    ignore_all: bool,
    valid: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """(target, mask) at stride OUTPUT_STRIDE for a crop of `size` (w, h) input pixels.

    Boxes are already in crop coordinates. `valid` is the part of the crop
    covered by the page (x0, y0, x1, y1); the rest is padding.
    """
    w, h = size[0] // OUTPUT_STRIDE, size[1] // OUTPUT_STRIDE
    target = np.zeros((h, w), np.float32)
    mask = np.ones((h, w), np.float32)  # padding is background; the page is set below

    def cells(box: list[float], inset: float = 0.0) -> tuple[int, int, int, int] | None:
        x0 = (box[0] + inset) / OUTPUT_STRIDE
        y0 = (box[1] + inset) / OUTPUT_STRIDE
        x1 = (box[0] + box[2] - inset) / OUTPUT_STRIDE
        y1 = (box[1] + box[3] - inset) / OUTPUT_STRIDE
        if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
            return None
        a, b = max(0, math.floor(x0)), max(0, math.floor(y0))
        c, d = min(w, math.ceil(x1)), min(h, math.ceil(y1))
        return (a, b, c, d) if c > a and d > b else None

    vx0, vy0, vx1, vy1 = valid
    page = cells([vx0, vy0, vx1 - vx0, vy1 - vy0])
    if page and ignore_all:
        a, b, c, d = page
        mask[b:d, a:c] = 0.0
    for box in background or []:
        region = cells(box)
        if region:
            a, b, c, d = region
            mask[b:d, a:c] = 1.0
    for box in ignore:
        region = cells(box)
        if region:
            a, b, c, d = region
            mask[b:d, a:c] = 0.0
    # The ring between a box and its core is background: the map means "region
    # core", which detector.py's unshrink assumes. Text overrides ignore; a box too
    # small to keep a core is ignored whole. Rings first, then cores.
    cores = []
    for box in text:
        outer = cells(box)
        if outer is None:
            continue
        a, b, c, d = outer
        core = cells(box, shrink_offset(box[2], box[3]))
        if core is not None and min(core[2] - core[0], core[3] - core[1]) >= MIN_SHRUNK_SIDE:
            mask[b:d, a:c] = 0.0 if ignore_all else 1.0
            cores.append(core)
        else:
            mask[b:d, a:c] = 0.0
    for a, b, c, d in cores:
        target[b:d, a:c] = 1.0
        mask[b:d, a:c] = 1.0
    return target, mask


class TextMapDataset(Dataset):
    def __init__(self, cache: Path, records: list[dict], crop: int = 768, train: bool = True,
                 scale_range: tuple[float, float] = (0.6, 1.25)) -> None:
        self.cache = cache
        self.records = records
        self.crop = crop
        self.train = train
        self.scale_range = scale_range

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(self.cache / record["file"]) as image:
            image = image.convert("RGB")
        # Jitter around the inference scale.
        base = INPUT_SIZE / max(image.size)
        s = base * (random.uniform(*self.scale_range) if self.train else 1.0)
        size = (max(1, round(image.width * s)), max(1, round(image.height * s)))
        image = image.resize(size, Image.BILINEAR)
        sx, sy = size[0] / record["width"], size[1] / record["height"]

        crop = self.crop if self.train else INPUT_SIZE
        if self.train:
            # Prefer crops that hold text: half the time centre on a random box.
            if record["text"] and random.random() < 0.5:
                box = random.choice(record["text"])
                cx = (box[0] + box[2] / 2) * sx + random.uniform(-crop / 3, crop / 3)
                cy = (box[1] + box[3] / 2) * sy + random.uniform(-crop / 3, crop / 3)
                x0, y0 = int(cx - crop / 2), int(cy - crop / 2)
            else:
                x0 = random.randint(min(0, size[0] - crop), max(0, size[0] - crop))
                y0 = random.randint(min(0, size[1] - crop), max(0, size[1] - crop))
        else:
            x0 = -(crop - size[0]) // 2 if size[0] < crop else 0
            y0 = -(crop - size[1]) // 2 if size[1] < crop else 0

        canvas = Image.new("RGB", (crop, crop), (PAD_VALUE,) * 3)
        canvas.paste(image, (-x0, -y0))

        def move(boxes: list[list[float]]) -> list[list[float]]:
            return [[b[0] * sx - x0, b[1] * sy - y0, b[2] * sx, b[3] * sy] for b in boxes]

        valid = (max(0, -x0), max(0, -y0), min(crop, size[0] - x0), min(crop, size[1] - y0))
        target, mask = paint_targets(
            (crop, crop), move(record["text"]), move(record.get("ignore", [])),
            move(record.get("background", [])), record.get("ignore_all", False), valid,
        )

        if self.train:
            canvas = augment(canvas)
        pixels = torch.from_numpy(np.asarray(canvas, np.float32) / 255.0).permute(2, 0, 1)
        return pixels, torch.from_numpy(target)[None], torch.from_numpy(mask)[None]


def augment(image: Image.Image) -> Image.Image:
    if random.random() < 0.3:
        image = ImageOps.grayscale(image).convert("RGB")
    if random.random() < 0.8:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.6, 1.4))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.15:
        image = ImageOps.invert(image)
    return image


def load_records(cache: Path) -> list[dict]:
    return json.loads((cache / "labels.json").read_text(encoding="utf-8"))
