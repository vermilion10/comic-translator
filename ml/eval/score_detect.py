"""Score the detect stage against hand-labelled comic pages.

Usage:
    pip install onnxruntime numpy pillow
    python ml/eval/score_detect.py                        # the shipped textseg
    python ml/eval/score_detect.py --model candidate.onnx # another export
    python ml/eval/score_detect.py --threshold 0.6        # the shipped default
    python ml/eval/score_detect.py --slice sfx --verbose  # every miss, listed

Reads detect_reference.json (post ids and boxes; pages download into
ml/eval/data/, gitignored).

The primary measure is coverage, the share of a region's area inside the
union of the detections, since the pipeline crops each box for OCR: a loose
box costs a little inpainting, a clipped one loses text. Best-IoU matching is
reported beside it. Results are split by kind (bubble, sfx, caption), dark vs
light background and language, with counts of merged and fragmented regions.
The reference is labelled exhaustively, so an unmatched detection is a false
positive.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from detect import (
    DEFAULT_IOU,
    DETECT_FLOOR,
    load_detector,
    area,
    encloses,
    intersection,
    iou,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = ROOT / "extension" / "public" / "models"
DANBOORU_CDN = "https://cdn.donmai.us/original"
USER_AGENT = "comic-translator-eval"

# A region counts as found when this much of it is covered; below half,
# OCR returns a fragment.
COVER_HIT = 0.5
# The conventional detection threshold, reported beside coverage.
IOU_HIT = 0.5
# Below this share inside any region a detection is spurious. Generous,
# because detector boxes are looser than the lettering.
PRECISION_HIT = 0.25

# A dark background: at least DARK_SHARE of the pixels below DARK_LUMA.
DARK_LUMA = 80
DARK_SHARE = 0.25

KINDS = ("bubble", "sfx", "caption")


def fetch_page(case: dict) -> Path:
    """By content hash, like score.py's Chinese and Korean sets."""
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{case['post']}.{case['ext']}"
    if not path.exists():
        digest = case["md5"]
        url = f"{DANBOORU_CDN}/{digest[:2]}/{digest[2:4]}/{digest}.{case['ext']}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=180) as response:
            path.write_bytes(response.read())
    return path


def dark_share(image: Image.Image, box: list[float]) -> float:
    x, y, width, height = (int(round(v)) for v in box)
    crop = image.convert("L").crop((x, y, x + max(1, width), y + max(1, height)))
    grey = np.asarray(crop, dtype=np.uint8)
    return float((grey < DARK_LUMA).mean()) if grey.size else 0.0


def covered(region: list[float], detections: list) -> float:
    """Share of `region` inside the union of `detections`, on a mask so overlaps are not counted twice."""
    x, y, width, height = region
    if width <= 0 or height <= 0:
        return 0.0
    overlapping = [d for d in detections if intersection(d.box, region) > 0]
    if not overlapping:
        return 0.0

    # Cap the grid so a huge region cannot allocate an enormous mask.
    columns = min(int(round(width)), 512) or 1
    rows = min(int(round(height)), 512) or 1
    xs = x + (np.arange(columns) + 0.5) * width / columns
    ys = y + (np.arange(rows) + 0.5) * height / rows
    mask = np.zeros((rows, columns), dtype=bool)
    for detection in overlapping:
        bx, by, bw, bh = detection.box
        inside_x = (xs >= bx) & (xs <= bx + bw)
        inside_y = (ys >= by) & (ys <= by + bh)
        mask |= inside_y[:, None] & inside_x[None, :]
    return float(mask.mean())


class Tally:
    __slots__ = ("count", "hits", "iou_hits", "coverage", "fragmented")

    def __init__(self) -> None:
        self.count = 0
        self.hits = 0
        self.iou_hits = 0
        self.coverage = 0.0
        self.fragmented = 0

    def add(self, coverage: float, best_iou: float, pieces: int) -> None:
        self.count += 1
        self.hits += coverage >= COVER_HIT
        self.iou_hits += best_iou >= IOU_HIT
        self.coverage += coverage
        self.fragmented += pieces > 1

    def line(self, label: str) -> str:
        if not self.count:
            return f"{label:14s}       no regions"
        return (
            f"{label:14s} {self.hits:4d}/{self.count:<4d} "
            f"{self.hits / self.count:6.3f}   "
            f"{self.coverage / self.count:6.3f}   "
            f"{self.iou_hits / self.count:6.3f}   "
            f"{self.fragmented:4d}"
        )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODELS / "textseg.onnx")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DETECT_FLOOR,
        help="confidence floor; the shipped default filter is 0.6 (settings.ts detectThreshold)",
    )
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="dedupe IoU")
    parser.add_argument(
        "--reference", type=Path, default=HERE / "detect_reference.json"
    )
    parser.add_argument(
        "--slice",
        choices=(*KINDS, "dark", "all"),
        default="all",
        help="with --verbose, which slice to list misses from",
    )
    parser.add_argument("--verbose", action="store_true", help="list every miss")
    args = parser.parse_args()

    if not args.model.exists():
        print(
            f"missing {args.model}\nrun 'npm run assets' in extension/ first",
            file=sys.stderr,
        )
        return 1
    if not args.reference.exists():
        print(f"missing {args.reference}", file=sys.stderr)
        return 1

    cases = json.loads(args.reference.read_text(encoding="utf-8"))
    detector = load_detector(args.model)

    by_kind: dict[str, Tally] = defaultdict(Tally)
    by_language: dict[str, Tally] = defaultdict(Tally)
    dark_tally, light_tally, overall = Tally(), Tally(), Tally()
    detections_total = spurious = merged = 0
    elapsed = 0.0

    for case in cases:
        page = fetch_page(case)
        image = Image.open(page)
        started = time.perf_counter()
        found = detector.detect(image, args.threshold, args.iou)
        elapsed += time.perf_counter() - started

        regions = case["regions"]
        detections_total += len(found)
        for detection in found:
            hit = sum(
                1
                for region in regions
                if encloses(detection.box, region["box"]) >= COVER_HIT
            )
            merged += hit > 1
            if not any(
                encloses(detection.box, region["box"]) >= PRECISION_HIT
                or encloses(region["box"], detection.box) >= PRECISION_HIT
                for region in regions
            ):
                spurious += 1

        misses = []
        for region in regions:
            box = region["box"]
            coverage = covered(box, found)
            best = max((iou(d.box, box) for d in found), default=0.0)
            pieces = sum(
                1 for d in found if intersection(d.box, box) / max(area(box), 1) > 0.05
            )
            dark = region.get("dark_share", dark_share(image, box)) >= DARK_SHARE

            for tally in (
                overall,
                by_kind[region["kind"]],
                by_language[case["language"]],
                dark_tally if dark else light_tally,
            ):
                tally.add(coverage, best, pieces)

            if coverage < COVER_HIT:
                wanted = args.slice
                if wanted == "all" or wanted == region["kind"] or (wanted == "dark" and dark):
                    misses.append((region, coverage, dark))

        print(
            f"{case['post']}  {case['language']}  "
            f"{len(regions):3d} regions  {len(found):3d} detections  "
            f"{len(regions) - sum(1 for r in regions if covered(r['box'], found) >= COVER_HIT):2d} missed"
        )
        if args.verbose:
            for region, coverage, dark in misses:
                x, y, w, h = (round(v) for v in region["box"])
                mark = " dark" if dark else ""
                note = f"  {region['note']}" if region.get("note") else ""
                print(
                    f"      miss {region['kind']:7s} {w:4d}x{h:<4d} at {x:5d},{y:<5d} "
                    f"coverage {coverage:.2f}{mark}{note}"
                )

    print(
        f"\nmodel {args.model.name} at threshold {args.threshold:.2f}"
        f"   {len(cases)} pages, {overall.count} regions"
        f"   {elapsed / max(1, len(cases)) * 1000:.0f} ms/page\n"
    )
    header = f"{'slice':14s} {'found':>9s} {'recall':>7s}  {'mean cov':>7s}  {'IoU>.5':>7s}  {'frag':>4s}"
    print(header)
    print("-" * len(header))
    for kind in KINDS:
        print(by_kind[kind].line(kind))
    print(dark_tally.line("dark bg"))
    print(light_tally.line("light bg"))
    for language in sorted(by_language):
        print(by_language[language].line(language))
    print("-" * len(header))
    print(overall.line("all"))

    print(
        f"\n{detections_total} detections, {spurious} spurious "
        f"({spurious / max(1, detections_total):.3f}), "
        f"{merged} covering two or more regions"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
