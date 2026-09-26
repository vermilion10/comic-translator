"""Write textseg probability maps and the Python decoder's output for a parity check.

    python ml/scripts/dump_text_map_fixture.py --model <textseg.onnx>
    node extension/scripts/check-text-map.mjs

Runs the model once per eval page and writes the raw map, the letterbox, the
boxes decoded under each decoder setting and the erase mask, so the Node side
can decode the same maps and compare. Writes ml/eval/data/text_map_fixture.json
(gitignored: derived from pages).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "eval"))
sys.path.insert(0, str(ROOT / "ml" / "textseg"))

import detector as textseg  # noqa: E402
from score_detect import fetch_page  # noqa: E402

# (BINARIZE, unshrink scale): the default decoder and a tighter one.
SETTINGS = [(0.3, 1.0), (0.2, 0.75)]
THRESHOLD = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "ml" / "eval" / "detect_reference.json")
    parser.add_argument("--out", type=Path, default=ROOT / "ml" / "eval" / "data" / "text_map_fixture.json")
    args = parser.parse_args()

    cases = json.loads(args.reference.read_text(encoding="utf-8"))
    model = textseg.TextMapDetector(args.model)
    original_binarize, original_unshrink = textseg.BINARIZE, textseg.unshrink
    pages = []
    try:
        for case in cases:
            prob, source, scale, pad_x, pad_y = model.infer(Image.open(fetch_page(case)))
            # The loop below patches the decoder; restore it for the mask on every page.
            textseg.BINARIZE, textseg.unshrink = original_binarize, original_unshrink
            mask = textseg.erase_mask(prob, source, scale, pad_x, pad_y, THRESHOLD)
            where = np.flatnonzero(mask)
            expected = []
            for binarize, unshrink_scale in SETTINGS:
                textseg.BINARIZE = binarize
                textseg.unshrink = lambda w, h, s=unshrink_scale: s * original_unshrink(w, h)
                found = textseg.TextMapDetector.decode(prob, source, scale, pad_x, pad_y, THRESHOLD)
                expected.append({"binarize": binarize, "unshrinkScale": unshrink_scale,
                                 "detections": [{"box": list(d.box), "score": d.score} for d in found]})
            pages.append({
                "eraseMask": {"pixels": int(where.size), "indexSum": int(where.sum())},
                "post": case["post"],
                "mapSize": int(prob.shape[0]),
                "prob": base64.b64encode(np.ascontiguousarray(prob, dtype="<f4").tobytes()).decode("ascii"),
                "layout": {"scale": scale, "padX": pad_x, "padY": pad_y},
                "width": source.width,
                "height": source.height,
                "expected": expected,
            })
    finally:
        textseg.BINARIZE, textseg.unshrink = original_binarize, original_unshrink

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"threshold": THRESHOLD, "pages": pages}), encoding="utf-8")
    print(f"{len(pages)} pages -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
