"""Write balloon-clip cases and ml/eval/balloon.py's answers for a parity check.

    python ml/scripts/dump_balloon_fixture.py
    node extension/scripts/check-balloon.mjs

Takes every STRIDE-th eval page, builds each region's erase mask and seed as
the extension does, and writes the clip window's pixels, masks and the Python
answer. Writes ml/eval/data/balloon_fixture.json (gitignored: cut from pages).
"""

from __future__ import annotations

import argparse
import base64
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
from score_detect import fetch_page  # noqa: E402

STRIDE = 5
THRESHOLD = 0.70


def box_mask(shape, box) -> np.ndarray:
    x, y, w, h = box
    mask = np.zeros(shape, bool)
    mask[int(max(0, y)):int(np.ceil(y + h)), int(max(0, x)):int(np.ceil(x + w))] = True
    return mask


def encode(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "extension" / "public" / "models" / "textseg.onnx")
    parser.add_argument("--out", type=Path, default=ROOT / "ml" / "eval" / "data" / "balloon_fixture.json")
    args = parser.parse_args()

    cases = json.loads((ROOT / "ml" / "eval" / "detect_reference.json").read_text(encoding="utf-8"))[::STRIDE]
    model = textseg.TextMapDetector(args.model)
    out = []
    for case in cases:
        image = Image.open(fetch_page(case)).convert("RGB")
        rgb = np.asarray(image)
        shape = rgb.shape[:2]
        regions = []
        prob, source, scale, pad_x, pad_y = model.infer(image)
        labels, count = ndimage.label(prob > textseg.BINARIZE)
        for i in range(1, count + 1):
            alone = np.where(labels == i, prob, 0).astype(prob.dtype)
            if model.decode(alone, source, scale, pad_x, pad_y, THRESHOLD):
                regions.append(("textseg",
                                textseg.erase_mask(alone, source, scale, pad_x, pad_y, THRESHOLD),
                                textseg.erase_mask(alone, source, scale, pad_x, pad_y, THRESHOLD, grow=0.0)))
        for detector, base, seed in regions:
            bounds = balloon.window_of(base)
            if bounds is None:
                continue
            y0, y1, x0, x1 = bounds
            answer = balloon.clip_to_balloon(rgb, base, seed)
            expected = None
            if answer is not None:
                where = np.flatnonzero(answer[0][y0:y1, x0:x1])
                expected = {"pixels": int(where.size), "indexSum": int(where.sum()), "color": list(answer[1]),
                            "lobes": [[x - x0, y - y0, w, h] for x, y, w, h in answer[2]]}
            out.append({
                "post": case["post"], "detector": detector, "width": x1 - x0, "height": y1 - y0,
                "rgb": encode(rgb[y0:y1, x0:x1]), "base": encode(base[y0:y1, x0:x1]),
                "seed": encode(seed[y0:y1, x0:x1]), "expected": expected,
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out), encoding="utf-8")
    filled = sum(c["expected"] is not None for c in out)
    print(f"{len(out)} regions from {len(cases)} pages ({filled} filled) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
