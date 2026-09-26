"""Score two detector exports on ml/eval/ and print the slices side by side, as deltas.

    python ml/scripts/compare_detect.py \
        --before extension/public/models/textseg.onnx \
        --after  ml/train/runs/finetune-v1/weights/best.onnx

Watch sfx and dark bg (the gaps), bubble (must not fall) and the spurious rate:
recall bought with false boxes costs an inpaint and a bad translation each.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORE = ROOT / "ml" / "eval" / "score_detect.py"

ROW = re.compile(
    r"^(?P<slice>\S[\S ]*?)\s+(?P<found>\d+)/(?P<total>\d+)\s+"
    r"(?P<recall>[\d.]+)\s+(?P<cov>[\d.]+)\s+(?P<iou>[\d.]+)\s+(?P<frag>\d+)\s*$"
)
SPURIOUS = re.compile(r"(\d+) detections, (\d+) spurious \(([\d.]+)\)")


def run(model: Path, threshold: float | None) -> tuple[dict, dict]:
    command = [sys.executable, str(SCORE), "--model", str(model)]
    if threshold is not None:
        command += ["--threshold", str(threshold)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(result.stdout[-2000:], result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"score_detect.py failed for {model}")

    slices, extra = {}, {}
    for line in result.stdout.splitlines():
        match = ROW.match(line)
        if match and match["slice"] not in ("slice",):
            slices[match["slice"].strip()] = {
                "found": int(match["found"]),
                "total": int(match["total"]),
                "recall": float(match["recall"]),
                "cov": float(match["cov"]),
                "iou": float(match["iou"]),
                "frag": int(match["frag"]),
            }
        spurious = SPURIOUS.search(line)
        if spurious:
            extra["detections"] = int(spurious.group(1))
            extra["spurious"] = int(spurious.group(2))
            extra["spurious_rate"] = float(spurious.group(3))
    return slices, extra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before", type=Path, default=ROOT / "extension" / "public" / "models" / "textseg.onnx"
    )
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"before: {args.before}")
    before, before_extra = run(args.before, args.threshold)
    print(f"after:  {args.after}")
    after, after_extra = run(args.after, args.threshold)

    order = [s for s in ("bubble", "sfx", "caption", "dark bg", "light bg", "ja", "ko", "zh", "all")
             if s in before or s in after]
    header = (f"{'slice':10s} {'n':>5s} | {'recall':>16s} | {'mean cov':>16s} | "
              f"{'IoU>.5':>16s}")
    print(f"\n{header}\n{'-' * len(header)}")
    for name in order:
        b, a = before.get(name), after.get(name)
        if not b or not a:
            continue
        cells = []
        for key in ("recall", "cov", "iou"):
            delta = a[key] - b[key]
            cells.append(f"{b[key]:.3f} -> {a[key]:.3f} {delta:+.3f}")
        mark = " *" if name in ("sfx", "dark bg") else "  "
        print(f"{name:10s}{mark}{b['total']:>3d} | " + " | ".join(f"{c:>16s}" for c in cells))
    print(f"{'-' * len(header)}\n* the two gaps this pass targets")

    if before_extra and after_extra:
        print(
            f"\nspurious: {before_extra['spurious']}/{before_extra['detections']} "
            f"({before_extra['spurious_rate']:.3f})  ->  "
            f"{after_extra['spurious']}/{after_extra['detections']} "
            f"({after_extra['spurious_rate']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
