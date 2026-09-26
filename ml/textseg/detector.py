"""The text-map model as a detector.

Map -> boxes, DBNet's way: binarise at BINARIZE, one region per connected
component, scored by its mean probability, its box grown back by the label
shrink. extension/src/pipeline/detect/text-map.ts mirrors decode(); change
one, change the other (extension/scripts/check-text-map.mjs checks them).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as rt
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from detect import DETECT_FLOOR, INPUT_SIZE, PAD_COLOR, Detection  # noqa: E402

OUTPUT_STRIDE = 4
SHRINK_RATIO = 0.4     # must match dataset.py
BINARIZE = 0.3
MIN_CELLS = 3          # components smaller than this many map cells are noise


def unshrink(w: float, h: float, ratio: float = SHRINK_RATIO) -> float:
    """The offset D dataset.py's shrink took off a (w+2D, h+2D) box, by fixed-point iteration."""
    k = 1 - ratio**2
    d = 0.0
    for _ in range(30):
        d = (w + 2 * d) * (h + 2 * d) * k / (2 * (w + h + 4 * d))
    return d


class TextMapDetector:
    def __init__(self, model: Path) -> None:
        options = rt.SessionOptions()
        options.log_severity_level = 3
        self.session = rt.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image: Image.Image, threshold: float = DETECT_FLOOR, iou_threshold: float = 0.5):
        return self.decode(*self.infer(image), threshold, iou_threshold)

    def infer(self, image: Image.Image):
        source = image.convert("RGB")
        scale = min(INPUT_SIZE / source.width, INPUT_SIZE / source.height)
        width, height = round(source.width * scale), round(source.height * scale)
        pad_x, pad_y = (INPUT_SIZE - width) // 2, (INPUT_SIZE - height) // 2
        canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), PAD_COLOR)
        canvas.paste(source.resize((width, height), Image.BILINEAR), (pad_x, pad_y))
        chw = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
        prob = self.session.run(None, {self.input_name: chw})[0][0, 0]
        return prob, source, scale, pad_x, pad_y

    @staticmethod
    def decode(prob: np.ndarray, source: Image.Image, scale: float, pad_x: int, pad_y: int,
               threshold: float = DETECT_FLOOR, iou_threshold: float = 0.5) -> list[Detection]:
        del iou_threshold  # components never overlap; there is nothing to dedupe
        labels, count = ndimage.label(prob > BINARIZE)
        if not count:
            return []
        index = np.arange(1, count + 1)
        scores = ndimage.mean(prob, labels, index)
        sizes = ndimage.sum(np.ones_like(prob), labels, index)
        detections = []
        for (rows, cols), score, cells in zip(ndimage.find_objects(labels), scores, sizes):
            if score < threshold or cells < MIN_CELLS:
                continue
            x0, x1 = cols.start * OUTPUT_STRIDE, cols.stop * OUTPUT_STRIDE
            y0, y1 = rows.start * OUTPUT_STRIDE, rows.stop * OUTPUT_STRIDE
            d = unshrink(x1 - x0, y1 - y0)
            x0, y0, x1, y1 = x0 - d, y0 - d, x1 + d, y1 + d
            left = max(0.0, min((x0 - pad_x) / scale, source.width))
            top = max(0.0, min((y0 - pad_y) / scale, source.height))
            right = max(0.0, min((x1 - pad_x) / scale, source.width))
            bottom = max(0.0, min((y1 - pad_y) / scale, source.height))
            if right > left and bottom > top:
                detections.append(Detection((left, top, right - left, bottom - top), float(score)))
        return sorted(detections, key=lambda d: -d.score)


# Mirrors ERASE_GROW in extension/src/pipeline/detect/text-map.ts.
ERASE_GROW = 0.75


def erase_mask(prob: np.ndarray, source: Image.Image, scale: float, pad_x: int, pad_y: int,
               threshold: float, grow: float = ERASE_GROW) -> np.ndarray:
    """The inpaint mask the extension builds from these detections' shapes, as a bool array.

    Mirrors text-map.ts's componentShape and mask.ts's paintShape down to the
    float operations, so check-text-map.mjs can require identical pixels.
    """
    width, height = source.width, source.height
    mask = np.zeros((height, width), bool)
    labels, count = ndimage.label(prob > BINARIZE)
    if not count:
        return mask
    index = np.arange(1, count + 1)
    scores = ndimage.mean(prob, labels, index)
    sizes = ndimage.sum(np.ones_like(prob), labels, index)
    cell = OUTPUT_STRIDE / scale
    for label, ((rows, cols), score, cells) in enumerate(zip(ndimage.find_objects(labels), scores, sizes), 1):
        if score < threshold or cells < MIN_CELLS:
            continue
        x0, x1 = cols.start * OUTPUT_STRIDE, cols.stop * OUTPUT_STRIDE
        y0, y1 = rows.start * OUTPUT_STRIDE, rows.stop * OUTPUT_STRIDE
        offset = unshrink(x1 - x0, y1 - y0)
        left = max(0.0, min((x0 - offset - pad_x) / scale, width))
        top = max(0.0, min((y0 - offset - pad_y) / scale, height))
        right = max(0.0, min((x1 + offset - pad_x) / scale, width))
        bottom = max(0.0, min((y1 + offset - pad_y) / scale, height))
        if not (right > left and bottom > top):
            continue
        wl, wt = max(0, int(np.floor(left))), max(0, int(np.floor(top)))
        wr = min(width, int(np.ceil(left + (right - left))))
        wb = min(height, int(np.ceil(top + (bottom - top))))
        if wr <= wl or wb <= wt:
            continue
        shape_x = (cols.start * OUTPUT_STRIDE - pad_x) / scale
        shape_y = (rows.start * OUTPUT_STRIDE - pad_y) / scale
        core_cells = labels[rows, cols] == label
        column = np.floor((np.arange(wl, wr) - shape_x) / cell).astype(int)
        row = np.floor((np.arange(wt, wb) - shape_y) / cell).astype(int)
        ok_c = (column >= 0) & (column < core_cells.shape[1])
        ok_r = (row >= 0) & (row < core_cells.shape[0])
        core = np.zeros((wb - wt, wr - wl), bool)
        core[np.ix_(ok_r, ok_c)] = core_cells[np.ix_(row[ok_r], column[ok_c])]
        if not core.any():
            continue
        # Squared distance in integers, so both sides compare exactly.
        _, (ny, nx) = ndimage.distance_transform_edt(~core, return_indices=True)
        yy, xx = np.indices(core.shape)
        squared = (yy - ny) ** 2 + (xx - nx) ** 2
        limit = (grow * offset) / scale
        mask[wt:wb, wl:wr] |= squared <= limit * limit
    return mask
