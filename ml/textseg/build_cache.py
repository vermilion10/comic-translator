"""Fetch the training pages once, downscale them, and write the text-map labels.

Usage (locally, or as a CPU-only Kaggle notebook):
    python ml/textseg/build_cache.py --out ml/textseg/cache
    python ml/textseg/build_cache.py --out /kaggle/working/cache --limit 20   # smoke

Writes `pages/<post>.jpg` at LONG_SIDE and `labels.json`, built once and
reused by every training run.

Each pixel is one of:
  text        a region in detect_training.json
  ignore      not in the loss: ignore_regions.json, likely text the automatic
              labels left out, so it is not taught as background
  background  everything else
Hard-negative pages are ignore everywhere except their confirmed text-free
windows, which are background.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "eval"))

import score_detect  # noqa: E402

TRAIN = ROOT / "ml" / "train"
REFERENCE = TRAIN / "detect_training.json"
IGNORE = TRAIN / "ignore_regions.json"
NEGATIVES = TRAIN / "hard_negatives.json"
# Every post id of the evaluation set; none may be trained on.
HELD_OUT = ROOT / "ml" / "eval" / "held_out.json"

# Stored above the 1280 inference size so scale augmentation can zoom in.
LONG_SIDE = 1536
JPEG_QUALITY = 90
Image.MAX_IMAGE_PIXELS = None  # the sample runs to 79 megapixels


def fetch_and_shrink(entry: dict, pages: Path, raw: Path, keep_raw: bool) -> tuple[float, int, int]:
    """Returns (scale applied, stored width, stored height)."""
    target = pages / f"{entry['post']}.jpg"
    if target.exists():
        with Image.open(target) as image:
            return image.width / entry["width"], image.width, image.height

    score_detect.DATA = raw
    for attempt in range(6):
        try:
            source = score_detect.fetch_page(entry)
            break
        except Exception as error:  # noqa: BLE001
            if attempt == 5:
                raise
            time.sleep(min(60, 3 * 2**attempt))
            print(f"  {entry['post']} {error}; retry {attempt + 1}", flush=True)

    with Image.open(source) as image:
        image = image.convert("RGB")
        scale = min(1.0, LONG_SIDE / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.LANCZOS,
            )
        image.save(target, quality=JPEG_QUALITY)
        size = image.size
    if not keep_raw:
        source.unlink(missing_ok=True)
    # Scale against the recorded page size, which the boxes are in.
    return size[0] / entry["width"], size[0], size[1]


def scaled(boxes: list[list[float]], s: float) -> list[list[float]]:
    return [[round(v * s, 1) for v in box[:4]] for box in boxes]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "ml" / "textseg" / "cache")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--ignore", type=Path, default=IGNORE)
    parser.add_argument("--negatives", type=Path, default=NEGATIVES)
    parser.add_argument("--workers", type=int, default=2,
                        help="Danbooru answers 429 at ~6 pages/s from one host; 2 stays under it")
    parser.add_argument("--limit", type=int, default=0, help="first N pages only, for a smoke run")
    parser.add_argument("--keep-raw", action="store_true", help="keep the full-size originals")
    args = parser.parse_args()

    pages_dir = args.out / "pages"
    raw_dir = args.out / "raw"
    pages_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    ignore = json.loads(args.ignore.read_text(encoding="utf-8"))
    negatives = json.loads(args.negatives.read_text(encoding="utf-8"))

    # The eval set must never be trained on.
    forbidden = set(json.loads(HELD_OUT.read_text(encoding="utf-8")))
    leaked = {p["post"] for p in reference + negatives} & forbidden
    assert not leaked, f"eval pages in the training input: {sorted(leaked)}"

    jobs = [("page", p) for p in reference] + [("negative", p) for p in negatives]
    if args.limit:
        jobs = jobs[: args.limit] + [j for j in jobs if j[0] == "negative"][: max(1, args.limit // 10)]

    records, failed = [], []
    started = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch_and_shrink, entry, pages_dir, raw_dir, args.keep_raw): (kind, entry)
                   for kind, entry in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            kind, entry = futures[future]
            try:
                s, width, height = future.result()
            except Exception as error:  # noqa: BLE001
                failed.append(entry["post"])
                print(f"  {entry['post']} gave up: {error}", flush=True)
                continue
            record = {"post": entry["post"], "file": f"pages/{entry['post']}.jpg",
                      "width": width, "height": height}
            if kind == "page":
                record.update(
                    kind="page",
                    reviewed=bool(entry.get("reviewed")),
                    text=scaled([r["box"] for r in entry["regions"]], s),
                    ignore=scaled(ignore.get(str(entry["post"]), []), s),
                )
            else:
                record.update(
                    kind="negative", reviewed=True, text=[], ignore_all=True,
                    background=scaled([c["window"] for c in entry["crops"]], s),
                )
            records.append(record)
            if done % 100 == 0 or done == len(jobs):
                rate = done / (time.perf_counter() - started)
                print(f"{done}/{len(jobs)} pages, {rate:.1f}/s", flush=True)

    records.sort(key=lambda r: (r["kind"], r["post"]))
    (args.out / "labels.json").write_text(json.dumps(records), encoding="utf-8")
    pages = [r for r in records if r["kind"] == "page"]
    print(f"{len(pages)} pages ({sum(r['reviewed'] for r in pages)} reviewed), "
          f"{len(records) - len(pages)} negative pages, "
          f"{sum(len(r['text']) for r in pages)} text regions, "
          f"{sum(len(r['ignore']) for r in pages)} ignore regions, {len(failed)} failed")
    return 1 if failed and len(failed) > 0.02 * len(jobs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
