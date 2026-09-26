"""Sweep detector exports across thresholds and compare them at matched spurious rate.

    python ml/scripts/sweep_detect.py \
        --model shipped=extension/public/models/textseg.onnx \
        --model candidate=ml/train/runs/<run>/textseg.onnx \
        --out ml/train/sweep.json

Models put their confidence on different scales, so they are compared at a
given spurious rate: for each ceiling, the best sfx and dark-background recall
each model reaches with its own threshold. Metrics come from score_detect.py.
Each page runs once per model and is re-decoded at every threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "eval"))

from detect import DEFAULT_IOU, area, load_detector, encloses, intersection, iou  # noqa: E402
from score_detect import (  # noqa: E402
    COVER_HIT,
    DARK_SHARE,
    IOU_HIT,
    KINDS,
    PRECISION_HIT,
    covered,
    dark_share,
    fetch_page,
)

# 0.05, then 0.01 steps from 0.10 to 0.95, fine enough near every model's gate.
DEFAULT_THRESHOLDS = sorted({0.05} | {round(0.01 * i, 2) for i in range(10, 96)})
# Spurious-rate ceilings compared at.
DEFAULT_CEILINGS = (0.009, 0.019, 0.028, 0.035, 0.043, 0.07)

SLICES = (*KINDS, "dark bg", "light bg", "all")


def score(pages: list[dict], detections_by_page: list[list]) -> dict:
    """One row of the sweep: score_detect.py's tallies for one threshold."""
    hits = defaultdict(int)
    iou_hits = defaultdict(int)
    coverage = defaultdict(float)
    count = defaultdict(int)
    fragmented = defaultdict(int)
    true_split = defaultdict(int)
    detections_total = spurious = 0

    for page, found in zip(pages, detections_by_page):
        regions = page["regions"]
        detections_total += len(found)
        for detection in found:
            if not any(
                encloses(detection.box, r["box"]) >= PRECISION_HIT
                or encloses(r["box"], detection.box) >= PRECISION_HIT
                for r in regions
            ):
                spurious += 1
        for region in regions:
            box = region["box"]
            cov = covered(box, found)
            best = max((iou(d.box, box) for d in found), default=0.0)
            pieces = sum(1 for d in found if intersection(d.box, box) / max(area(box), 1) > 0.05)
            # A region fragmented only because a neighbour clips its edge is a spurious
            # problem; a true split is one no single box covers.
            single = max((encloses(d.box, box) for d in found), default=0.0)
            names = ("all", region["kind"], "dark bg" if region["dark"] else "light bg")
            for name in names:
                count[name] += 1
                hits[name] += cov >= COVER_HIT
                iou_hits[name] += best >= IOU_HIT
                coverage[name] += cov
                fragmented[name] += pieces > 1
                true_split[name] += pieces > 1 and single < 0.9

    row = {"detections": detections_total, "spurious": spurious,
           "spurious_rate": spurious / max(1, detections_total)}
    for name in SLICES:
        n = max(1, count[name])
        row[name] = {"n": count[name], "recall": hits[name] / n, "cov": coverage[name] / n,
                     "iou": iou_hits[name] / n, "frag": fragmented[name],
                     "true_split": true_split[name]}
    return row


def sweep(model: Path, pages: list[dict], images: list[Image.Image],
          thresholds: list[float]) -> dict[str, dict]:
    detector = load_detector(model)
    started = time.perf_counter()
    raw = [detector.infer(image) for image in images]
    print(f"  {model.name}: {len(images)} pages inferred in {time.perf_counter() - started:.0f}s",
          file=sys.stderr)
    out = {}
    for threshold in thresholds:
        found = [detector.decode(*r, threshold, DEFAULT_IOU) for r in raw]
        out[f"{threshold:.2f}"] = score(pages, found)
    return out


def best_under(rows: dict[str, dict], ceiling: float) -> tuple[str, dict] | None:
    """The threshold with the best sfx recall whose spurious rate fits the ceiling."""
    fitting = [(t, r) for t, r in rows.items() if r["spurious_rate"] <= ceiling + 1e-9]
    if not fitting:
        return None
    return max(fitting, key=lambda tr: (tr[1]["sfx"]["recall"], tr[1]["dark bg"]["recall"]))


def print_fixed(results: dict[str, dict], threshold: float) -> None:
    key = f"{threshold:.2f}"
    names = list(results)
    print(f"\nat threshold {key}")
    header = f"{'slice':10s} | " + " | ".join(f"{n:>18s}" for n in names)
    print(header)
    print("-" * len(header))
    for s in SLICES:
        cells = []
        for n in names:
            r = results[n][key][s]
            cells.append(f"{r['recall']:.3f} {r['cov']:.3f} {r['iou']:.3f}")
        print(f"{s:10s} | " + " | ".join(f"{c:>18s}" for c in cells))
    print(f"{'spurious':10s} | " + " | ".join(
        f"{results[n][key]['spurious']:>3d}/{results[n][key]['detections']:<4d} {results[n][key]['spurious_rate']:.3f}".rjust(18)
        for n in names))
    print(f"{'frag/split':10s} | " + " | ".join(
        f"{results[n][key]['all']['frag']:>3d} / {results[n][key]['all']['true_split']:<3d}".rjust(18) for n in names))
    print("(cells: recall  mean-coverage  IoU>.5)")


def print_matched(results: dict[str, dict], ceilings: tuple[float, ...]) -> None:
    names = list(results)
    print("\nbest sfx recall under a spurious ceiling (th - sfx / dark / all)")
    header = f"{'ceiling':8s} | " + " | ".join(f"{n:>26s}" for n in names) + " | better on sfx+dark"
    print(header)
    print("-" * len(header))
    for ceiling in ceilings:
        cells, picks = [], {}
        for n in names:
            pick = best_under(results[n], ceiling)
            if pick is None:
                cells.append("none fits")
                continue
            t, r = pick
            picks[n] = (r["sfx"]["recall"], r["dark bg"]["recall"])
            cells.append(f"th {t} - {r['sfx']['recall']:.3f} / {r['dark bg']['recall']:.3f} / {r['all']['recall']:.3f}")
        winners = [n for n in picks if all(
            picks[n][0] >= picks[m][0] and picks[n][1] >= picks[m][1] for m in picks)]
        verdict = winners[0] if len(winners) == 1 else ("split" if picks else "-")
        print(f"{ceiling:8.3f} | " + " | ".join(f"{c:>26s}" for c in cells) + f" | {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True,
                        help="name=path.onnx; repeatable, printed in the order given")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "ml" / "eval" / "detect_reference.json")
    parser.add_argument("--out", type=Path, default=None, help="write the full sweep as JSON")
    parser.add_argument("--load", type=Path, default=None,
                        help="a previous --out file; models already in it are not re-run")
    parser.add_argument("--ceiling", type=float, action="append", default=None)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    models = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        assert path, f"--model wants name=path, got {spec!r}"
        models.append((name, Path(path)))

    results: dict[str, dict] = {}
    if args.load and args.load.exists():
        results = json.loads(args.load.read_text(encoding="utf-8"))

    todo = [(n, p) for n, p in models if n not in results]
    if todo:
        cases = json.loads(args.reference.read_text(encoding="utf-8"))
        images = [Image.open(fetch_page(c)) for c in cases]
        pages = []
        for case, image in zip(cases, images):
            regions = []
            for region in case["regions"]:
                dark = region.get("dark_share", dark_share(image, region["box"])) >= DARK_SHARE
                regions.append({"box": region["box"], "kind": region["kind"], "dark": dark})
            pages.append({"regions": regions})
        for name, path in todo:
            assert path.exists(), f"missing {path}"
            results[name] = sweep(path, pages, images, DEFAULT_THRESHOLDS)

    results = {n: results[n] for n, _ in models}
    if args.out:
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print_fixed(results, 0.25)
    print_fixed(results, 0.05)
    print_matched(results, tuple(args.ceiling) if args.ceiling else DEFAULT_CEILINGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
