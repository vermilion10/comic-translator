"""How much lettering an erase mask takes, against how much of everything else.

    python ml/scripts/measure_erase_mask.py [--grows 0.5 0.6 0.75 1.0]

Measured on the 57 eval pages against the hand-drawn reference boxes:

  letters erased     share of the letters' dark pixels (luma < 128) the mask
                     covers, in light-background regions it touches at all. A
                     letter is a dark shape staying inside its box grown by
                     4 px, so outlines crossing the box do not count.
  other ink erased   dark pixels erased outside every reference box grown by
                     GROW px, per page

Rows: textseg boxes, textseg erase shapes at each grow factor, and the shapes
clipped to their balloons (`filled` counts balloon-filled regions).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "eval"))
sys.path.insert(0, str(ROOT / "ml" / "textseg"))

import balloon  # noqa: E402
import detector as textseg  # noqa: E402
from score_detect import DARK_SHARE, dark_share, fetch_page  # noqa: E402

GROW = 3
THRESHOLD = 0.70


def box_mask(shape: tuple[int, int], boxes) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for x, y, w, h in boxes:
        mask[int(max(0, y)):int(np.ceil(y + h)), int(max(0, x)):int(np.ceil(x + w))] = True
    return mask


def letters(dark: np.ndarray, shapes: np.ndarray, region: np.ndarray) -> np.ndarray:
    grown = ndimage.binary_dilation(region, iterations=4)
    rim = ndimage.binary_dilation(grown) & ~grown
    crossing = np.unique(shapes[dark & rim])
    return dark & region & ~np.isin(shapes, crossing[crossing > 0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "extension" / "public" / "models" / "textseg.onnx")
    parser.add_argument("--grows", type=float, nargs="+", default=[0.5, 0.6, 0.75, 1.0])
    args = parser.parse_args()

    cases = json.loads((ROOT / "ml" / "eval" / "detect_reference.json").read_text(encoding="utf-8"))
    model = textseg.TextMapDetector(args.model)
    shape_rows = [f"textseg shape x{g} @{THRESHOLD}" for g in args.grows]
    balloon_rows = [f"textseg shape x{textseg.ERASE_GROW} + balloon"]
    rows = [f"textseg box @{THRESHOLD}", *shape_rows, *balloon_rows]
    tally = {row: {"hit": 0, "ink": 0, "other": 0, "found": 0, "filled": 0, "regions": 0} for row in rows}

    for case in cases:
        image = Image.open(fetch_page(case)).convert("RGB")
        shape = (image.height, image.width)
        rgb = np.asarray(image)
        dark = np.asarray(image.convert("L")) < 128
        shapes, _ = ndimage.label(dark)
        light = [r for r in case["regions"]
                 if r.get("dark_share", dark_share(image, r["box"])) < DARK_SHARE]
        outside = ~ndimage.binary_dilation(box_mask(shape, [r["box"] for r in case["regions"]]),
                                           iterations=GROW)
        prob, source, scale, pad_x, pad_y = model.infer(image)
        masks = {
            rows[0]: box_mask(shape, [d.box for d in model.decode(prob, source, scale, pad_x, pad_y, THRESHOLD)]),
        }
        for g, row in zip(args.grows, shape_rows):
            masks[row] = textseg.erase_mask(prob, source, scale, pad_x, pad_y, THRESHOLD, grow=g)

        # The balloon rows go region by region, as the extension clips them.
        regions = {balloon_rows[0]: []}
        labels, count = ndimage.label(prob > textseg.BINARIZE)
        for i in range(1, count + 1):
            alone = np.where(labels == i, prob, 0).astype(prob.dtype)
            if model.decode(alone, source, scale, pad_x, pad_y, THRESHOLD):
                regions[balloon_rows[0]].append((
                    textseg.erase_mask(alone, source, scale, pad_x, pad_y, THRESHOLD),
                    textseg.erase_mask(alone, source, scale, pad_x, pad_y, THRESHOLD, grow=0.0),
                ))
        for row, pairs in regions.items():
            mask = np.zeros(shape, bool)
            for base, seed in pairs:
                clipped = balloon.clip_to_balloon(rgb, base, seed)
                mask |= base if clipped is None else clipped[0]
                tally[row]["filled"] += clipped is not None
                tally[row]["regions"] += 1
            masks[row] = mask

        for row, mask in masks.items():
            tally[row]["other"] += int((mask & dark & outside).sum())
            for region in light:
                inside = box_mask(shape, [region["box"]])
                if not (mask & inside).any():
                    continue
                target = letters(dark, shapes, inside)
                tally[row]["found"] += 1
                tally[row]["hit"] += int((mask & target).sum())
                tally[row]["ink"] += int(target.sum())

    print(f"{'mask':32s} regions  letters erased  other ink erased (px/page)  filled")
    for row in rows:
        t = tally[row]
        filled = f"{t['filled']}/{t['regions']}" if t["regions"] else ""
        print(f"{row:32s} {t['found']:7d}  {t['hit'] / max(1, t['ink']):14.3f}  "
              f"{t['other'] / len(cases):26.0f}  {filled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
