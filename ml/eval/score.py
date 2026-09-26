"""Score an OCR stage against a fixed reference set.

Usage:
    pip install onnxruntime numpy pillow
    python ml/eval/score.py                       # manga-ocr, Japanese
    python ml/eval/score.py --pages               # manga-ocr, multi-column balloons
    python ml/eval/score.py --engine paddle       # PP-OCR, Chinese
    python ml/eval/score.py --engine paddle --language ko   # PP-OCR, Korean

Scores the weights the extension ships. Japanese: kha-white/manga-ocr's 12
test crops, or with --pages 24 hand-transcribed multi-column balloons
(ja_page_reference.json). Chinese and Korean: zh_reference.json and
ko_reference.json, hand transcriptions. The references record post ids and
boxes, not pixels; pages download into ml/eval/data/ (gitignored).

The decode paths mirror extension/src/pipeline/ocr/; change one, change the other.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as rt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = ROOT / "extension" / "public" / "models"
REFERENCE = "https://raw.githubusercontent.com/kha-white/manga-ocr/master/tests/data"

INPUT_SIZE = 224
START_TOKEN, EOS_TOKEN, MAX_LENGTH = 2, 3, 300
FIRST_TEXT_TOKEN = 5
MAX_REPEATS = 16

# PP-OCR, mirroring extension/src/pipeline/ocr/paddle-ocr.ts and db-boxes.ts.
REC_HEIGHT = 48
DET_LIMIT, DET_STRIDE = 960, 32
REC_WIDTH_STEP, MIN_REC_WIDTH = 8, 160
VERTICAL_ASPECT = 1.5
BINARY_THRESHOLD, BOX_THRESHOLD, UNCLIP_RATIO, MIN_SIDE = 0.3, 0.6, 1.5, 3
INK_ACTIVE_FRACTION, INK_GAP_FRACTION, MIN_CELL_FRACTION = 0.03, 0.045, 0.3
DET_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
DET_STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)
DANBOORU_CDN = "https://cdn.donmai.us/original"


def fetch_reference() -> list[dict[str, str]]:
    DATA.mkdir(parents=True, exist_ok=True)
    expected_path = DATA / "expected_results.json"
    if not expected_path.exists():
        urllib.request.urlretrieve(f"{REFERENCE}/expected_results.json", expected_path)

    cases = json.loads(expected_path.read_text(encoding="utf-8"))
    for case in cases:
        image = DATA / case["filename"]
        if not image.exists():
            urllib.request.urlretrieve(f"{REFERENCE}/images/{case['filename']}", image)
    return cases


def post_process(text: str) -> str:
    """Port of manga_ocr.ocr.post_process, matching postprocess.ts."""
    text = "".join(text.split()).replace("…", "...")
    text = re.sub(r"[・.]{2,}", lambda m: "." * (m.end() - m.start()), text)
    return "".join(
        chr(ord(c) + 0xFEE0) if 0x21 <= ord(c) <= 0x7E else c for c in text
    )


def preprocess(source: Path | Image.Image) -> np.ndarray:
    # Greyscale, squashed to a square as the model was trained.
    image = (source if isinstance(source, Image.Image) else Image.open(source)).convert("L").convert("RGB")
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return ((array - 0.5) / 0.5).transpose(2, 0, 1)[None]


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


class Recognizer:
    def __init__(self, encoder: Path, decoder: Path, vocab: Path) -> None:
        options = rt.SessionOptions()
        options.log_severity_level = 3
        # 'basic', as in manga-ocr.ts.
        options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.encoder = rt.InferenceSession(
            str(encoder), options, providers=["CPUExecutionProvider"]
        )
        self.decoder = rt.InferenceSession(
            str(decoder), options, providers=["CPUExecutionProvider"]
        )
        self.vocab = vocab.read_text(encoding="utf-8").split("\n")
        if self.vocab and self.vocab[-1] == "":
            self.vocab.pop()

    def read(self, path: Path | Image.Image) -> str:
        hidden = self.encoder.run(None, {"pixel_values": preprocess(path)})[0]
        ids = [START_TOKEN]
        repeats = 0
        for _ in range(MAX_LENGTH):
            logits = self.decoder.run(
                None,
                {
                    "input_ids": np.array([ids], dtype=np.int64),
                    "encoder_hidden_states": hidden,
                },
            )[0][0, -1]
            best = int(np.argmax(logits))
            if best == EOS_TOKEN:
                break
            repeats = repeats + 1 if best == ids[-1] else 0
            if repeats >= MAX_REPEATS:
                break
            ids.append(best)

        text = "".join(
            self.vocab[i]
            for i in ids[1:]
            if i >= FIRST_TEXT_TOKEN and not self.vocab[i].startswith("<unused")
        )
        return post_process(text)



# Mirrors manga-ocr.ts: big multi-column regions are read in column chunks.
CHUNK_CHARS = 18
VERTICAL_SHARE = 0.6


def chunk_boxes(lines: list[tuple[float, float, float, float]]) -> list[tuple[int, int, int, int]] | None:
    """(x0, y0, x1, y1) crops, right to left, or None to read the region whole."""
    vertical = [line for line in lines if line[3] > VERTICAL_ASPECT * line[2]]
    if len(lines) < 2 or len(vertical) < VERTICAL_SHARE * len(lines):
        return None
    columns = sorted(lines, key=lambda b: -(b[0] + b[2] / 2))
    groups, current, chars = [], [], 0.0
    for column in columns:
        estimate = column[3] / max(1, column[2])
        if current and chars + estimate > CHUNK_CHARS:
            groups.append(current)
            current, chars = [], 0.0
        current.append(column)
        chars += estimate
    groups.append(current)
    if len(groups) == 1:
        return None
    return [
        (
            int(min(c[0] for c in g)),
            int(min(c[1] for c in g)),
            int(max(c[0] + c[2] for c in g)),
            int(max(c[1] + c[3] for c in g)),
        )
        for g in groups
    ]


def read_region(recognizer: "Recognizer", finder: "PaddleRecognizer", region: Image.Image) -> str:
    """manga-ocr over one region, in column chunks when it is big (manga-ocr.ts)."""
    chunks = chunk_boxes(finder._lines(region))
    if chunks is None:
        return recognizer.read(region)
    return "".join(recognizer.read(region.crop(chunk)) for chunk in chunks)


def fetch_paddle_reference(language: str) -> list[dict]:
    """Crops from real comic pages, fetched by content hash."""
    DATA.mkdir(parents=True, exist_ok=True)
    cases = json.loads(
        (Path(__file__).resolve().parent / f"{language}_reference.json").read_text(
            encoding="utf-8"
        )
    )
    for case in cases:
        page = DATA / f"{case['post']}.{case['ext']}"
        if not page.exists():
            digest = case["md5"]
            url = f"{DANBOORU_CDN}/{digest[:2]}/{digest[2:4]}/{digest}.{case['ext']}"
            request = urllib.request.Request(url, headers={"User-Agent": "comic-translator-eval"})
            with urllib.request.urlopen(request, timeout=120) as response:
                page.write_bytes(response.read())
    return cases


def lines_from_probability_map(
    probability: np.ndarray, scale_x: float, scale_y: float
) -> list[tuple[float, float, float, float]]:
    """Mirror of db-boxes.ts: connected components of the thresholded map, grown back by the unclip."""
    height, width = probability.shape
    flat = probability.ravel()
    seen = np.zeros(flat.size, dtype=bool)
    boxes = []
    for start in range(flat.size):
        if seen[start] or flat[start] <= BINARY_THRESHOLD:
            continue
        left, top, right, bottom = width, height, -1, -1
        total = count = 0
        seen[start] = True
        stack = [start]
        while stack:
            index = stack.pop()
            x, y = index % width, index // width
            left, right = min(left, x), max(right, x)
            top, bottom = min(top, y), max(bottom, y)
            total += flat[index]
            count += 1
            for neighbour, inside in (
                (index - 1, x > 0),
                (index + 1, x + 1 < width),
                (index - width, y > 0),
                (index + width, y + 1 < height),
            ):
                if inside and not seen[neighbour] and flat[neighbour] > BINARY_THRESHOLD:
                    seen[neighbour] = True
                    stack.append(neighbour)

        box_width, box_height = right - left + 1, bottom - top + 1
        if box_width < MIN_SIDE or box_height < MIN_SIDE:
            continue
        if total / count < BOX_THRESHOLD:
            continue
        grow = box_width * box_height * UNCLIP_RATIO / (2 * (box_width + box_height))
        boxes.append(
            (
                (left - grow) * scale_x,
                (top - grow) * scale_y,
                (box_width + 2 * grow) * scale_x,
                (box_height + 2 * grow) * scale_y,
            )
        )
    return boxes


def in_reading_order(
    boxes: list[tuple[float, float, float, float]], vertical: bool
) -> list[tuple[float, float, float, float]]:
    """Mirror of reading-order.ts: group boxes into lines by centre, then order them."""
    if not boxes:
        return []
    if vertical:
        centre, thickness, along = (
            lambda b: b[0] + b[2] / 2,
            lambda b: b[2],
            lambda b: b[1],
        )
    else:
        centre, thickness, along = (
            lambda b: b[1] + b[3] / 2,
            lambda b: b[3],
            lambda b: b[0],
        )

    rows: list[dict] = []
    for box in sorted(boxes, key=centre):
        for row in rows:
            if abs(centre(box) - row["centre"]) < 0.5 * min(thickness(box), row["thickness"]):
                row["boxes"].append(box)
                row["thickness"] = min(row["thickness"], thickness(box))
                break
        else:
            rows.append({"centre": centre(box), "thickness": thickness(box), "boxes": [box]})

    # Vertical CJK reads right to left, so its columns come in reverse.
    if vertical:
        rows.reverse()
    return [box for row in rows for box in sorted(row["boxes"], key=along)]


class PaddleRecognizer:
    """Mirror of paddle-ocr.ts: find the lines in a region, then read each one."""

    def __init__(
        self, det: Path, rec: Path, dictionary: Path, line_separator: str = ""
    ) -> None:
        self.line_separator = line_separator
        options = rt.SessionOptions()
        options.log_severity_level = 3
        self.det = rt.InferenceSession(str(det), options, providers=["CPUExecutionProvider"])
        self.rec = rt.InferenceSession(str(rec), options, providers=["CPUExecutionProvider"])
        entries = dictionary.read_text(encoding="utf-8").split("\n")
        if entries and entries[-1] == "":
            entries.pop()
        # Class 0 is the CTC blank and the last class is a space.
        self.charset = [""] + entries + [" "]

    def _lines(self, region: Image.Image) -> list[tuple[float, float, float, float]]:
        scale = min(1.0, DET_LIMIT / max(region.width, region.height))

        def snap(value: float) -> int:
            return max(DET_STRIDE, round(value * scale / DET_STRIDE) * DET_STRIDE)

        width, height = snap(region.width), snap(region.height)
        array = np.asarray(region.resize((width, height), Image.BILINEAR), dtype=np.float32)
        array = (array[:, :, ::-1] / 255.0 - DET_MEAN) / DET_STD
        outputs = self.det.run(
            None, {self.det.get_inputs()[0].name: array.transpose(2, 0, 1)[None]}
        )
        return lines_from_probability_map(
            outputs[0][0, 0], region.width / width, region.height / height
        )

    def _read_line(self, line: Image.Image) -> str:
        aspect = line.width / max(1, line.height)
        padded = max(
            MIN_REC_WIDTH,
            int(np.ceil(REC_HEIGHT * aspect / REC_WIDTH_STEP) * REC_WIDTH_STEP),
        )
        drawn = max(1, min(padded, round(REC_HEIGHT * aspect)))
        strip = np.zeros((REC_HEIGHT, padded, 3), dtype=np.float32)
        resized = np.asarray(line.resize((drawn, REC_HEIGHT), Image.BICUBIC), dtype=np.float32)
        strip[:, :drawn] = resized[:, :, ::-1] / 127.5 - 1
        outputs = self.rec.run(
            None, {self.rec.get_inputs()[0].name: strip.transpose(2, 0, 1)[None]}
        )[0][0]

        text, previous = [], -1
        for step in outputs:
            best = int(np.argmax(step))
            if best != previous and best != 0:
                text.append(self.charset[best])
            previous = best
        return "".join(text)

    @staticmethod
    def _ink_row_counts(column: Image.Image) -> np.ndarray:
        """Per-row ink count, with the ink polarity found by Otsu's method."""
        gray = np.asarray(column.convert("L"), dtype=np.float32)
        histogram, _ = np.histogram(gray, bins=256, range=(0, 256))
        total = gray.size
        sum_all = float(np.dot(np.arange(256), histogram))
        sum_below = 0.0
        weight_below = 0
        best_threshold, best_variance = 0, -1.0
        for value in range(256):
            weight_below += histogram[value]
            if weight_below == 0:
                continue
            weight_above = total - weight_below
            if weight_above == 0:
                break
            sum_below += value * histogram[value]
            mean_below = sum_below / weight_below
            mean_above = (sum_all - sum_below) / weight_above
            variance = weight_below * weight_above * (mean_below - mean_above) ** 2
            if variance > best_variance:
                best_variance, best_threshold = variance, value

        dark = gray <= best_threshold
        ink = dark if dark.mean() < 0.5 else ~dark
        return ink.sum(axis=1).astype(np.float64)

    @staticmethod
    def _ink_segments(counts: np.ndarray, width: int) -> list[tuple[int, int]]:
        """Row ranges to cut a column into at real ink gaps; the whole column when there are none."""
        active_threshold = max(1.0, width * INK_ACTIVE_FRACTION)
        min_gap = max(1, round(width * INK_GAP_FRACTION))
        min_cell = width * MIN_CELL_FRACTION

        raw: list[list[int]] = []
        start = -1
        gap = 0
        for row, count in enumerate(counts):
            if count > active_threshold:
                if start == -1:
                    start = row
                gap = 0
            elif start != -1:
                gap += 1
                if gap >= min_gap:
                    raw.append([start, row - gap + 1])
                    start = -1
                    gap = 0
        if start != -1:
            raw.append([start, len(counts)])

        segments = [(top, bottom) for top, bottom in raw if bottom - top >= min_cell]
        return segments or [(0, len(counts))]

    @classmethod
    def _unroll(cls, column: Image.Image) -> Image.Image:
        """A vertical column cut at ink gaps and laid out left to right."""
        side = column.width
        counts = cls._ink_row_counts(column)
        segments = cls._ink_segments(counts, side)
        strip = Image.new("RGB", (side * len(segments), side), (255, 255, 255))
        for cell, (top, bottom) in enumerate(segments):
            box = (0, top, side, bottom)
            strip.paste(column.crop(box).resize((side, side), Image.BICUBIC), (cell * side, 0))
        return strip

    def read(self, page: Path, box: list[float] | None = None) -> str:
        image = Image.open(page).convert("RGB")
        if box is not None:
            x, y, width, height = box
            image = image.crop((int(x), int(y), int(x + width), int(y + height)))

        lines = self._lines(image)
        if not lines:
            return self._read_line(image)

        tall = sum(1 for _, _, w, h in lines if h > w * VERTICAL_ASPECT)
        vertical = tall * 2 >= len(lines)

        parts = []
        for x, y, width, height in in_reading_order(lines, vertical):
            crop = image.crop(
                (
                    max(0, int(x)),
                    max(0, int(y)),
                    min(image.width, int(x + width)),
                    min(image.height, int(y + height)),
                )
            )
            if crop.width < 4 or crop.height < 4:
                continue
            if height > width * VERTICAL_ASPECT:
                crop = self._unroll(crop)
            text = self._read_line(crop)
            if text:
                parts.append(text)
        return self.line_separator.join(parts)


def score_paddle(args: argparse.Namespace) -> int:
    korean = args.language == "ko"
    rec = args.rec or (MODELS / ("ppocr-rec-ko.onnx" if korean else "ppocr-rec.onnx"))
    dictionary = args.dict or (MODELS / ("ppocr-dict-ko.txt" if korean else "ppocr-dict.txt"))
    for path in (args.det, rec, dictionary):
        if not path.exists():
            print(f"missing {path}\nrun 'npm run assets' in extension/ first", file=sys.stderr)
            return 1

    cases = fetch_paddle_reference(args.language)
    # Korean spaces its words; Chinese does not.
    recognizer = PaddleRecognizer(args.det, rec, dictionary, " " if korean else "")

    totals: dict[str, list[int]] = {"horizontal": [0, 0, 0, 0], "vertical": [0, 0, 0, 0]}
    for case in cases:
        started = time.perf_counter()
        got = recognizer.read(DATA / f"{case['post']}.{case['ext']}", case["box"])
        want = case["text"]
        bucket = totals[case["orientation"]]
        bucket[0] += got == want
        bucket[1] += 1
        bucket[2] += edit_distance(got, want)
        bucket[3] += len(want)

        elapsed = (time.perf_counter() - started) * 1000
        print(f"{case['post']}  {'OK  ' if got == want else 'DIFF'}  {elapsed:6.0f}ms  {got}")
        if got != want:
            print(f"{'':>24}expected  {want}")

    print()
    for orientation, (exact, count, distance, characters) in totals.items():
        if count:
            print(
                f"{orientation:11s} exact {exact}/{count}   "
                f"char error {distance / characters:.3f}"
            )
    return 0


def score_pages(read) -> int:
    """Score ja_page_reference.json. Ellipses, dots, hearts, notes and spaces are
    dropped and widths folded (NFKC) first, since transcription and model write
    them differently.
    """
    import unicodedata

    from score_detect import fetch_page

    def loose(text: str) -> str:
        return re.sub(r"[.．…・♪❤♥♡\s]", "", unicodedata.normalize("NFKC", text))

    pages = {c["post"]: c for c in json.loads((HERE / "detect_reference.json").read_text(encoding="utf-8"))}
    cases = json.loads((HERE / "ja_page_reference.json").read_text(encoding="utf-8"))
    exact = distance = characters = 0
    for case in cases:
        region = Image.open(fetch_page(pages[case["post"]])).convert("RGB").crop(case["box"])
        got, want = read(region), case["text"]
        errors = edit_distance(loose(got), loose(want))
        exact += errors == 0
        distance += errors
        characters += len(loose(want))
        print(f"{case['post']}  {'OK  ' if errors == 0 else 'DIFF'}  {got}")
        if errors:
            print(f"{'':>10}expected  {want}")
    print(f"\nexact match  {exact}/{len(cases)}\nchar error   {distance / characters:.3f}")
    return 0


def main() -> int:
    # Windows consoles default to a code page that cannot print CJK.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, default=MODELS / "manga-ocr-encoder.onnx")
    parser.add_argument("--decoder", type=Path, default=MODELS / "manga-ocr-decoder.onnx")
    parser.add_argument("--vocab", type=Path, default=MODELS / "manga-ocr-vocab.txt")
    parser.add_argument("--engine", choices=("manga-ocr", "paddle"), default="manga-ocr")
    parser.add_argument("--language", choices=("zh", "ko"), default="zh")
    parser.add_argument("--det", type=Path, default=MODELS / "ppocr-det.onnx")
    parser.add_argument("--rec", type=Path, default=None)
    parser.add_argument("--dict", type=Path, default=None)
    parser.add_argument("--pages", action="store_true",
                        help="Japanese: score the multi-column balloons of ja_page_reference.json")
    parser.add_argument("--whole", action="store_true",
                        help="Japanese: read every region whole, without column chunks")
    args = parser.parse_args()

    if args.engine == "paddle":
        return score_paddle(args)

    for path in (args.encoder, args.decoder, args.vocab):
        if not path.exists():
            print(f"missing {path}\nrun 'npm run assets' in extension/ first", file=sys.stderr)
            return 1

    recognizer = Recognizer(args.encoder, args.decoder, args.vocab)
    finder = PaddleRecognizer(args.det, MODELS / "ppocr-rec.onnx", MODELS / "ppocr-dict.txt")

    def read(image: Image.Image) -> str:
        return recognizer.read(image) if args.whole else read_region(recognizer, finder, image)

    if args.pages:
        return score_pages(read)

    cases = fetch_reference()

    exact = 0
    distance = characters = 0
    for case in cases:
        started = time.perf_counter()
        got = read(Image.open(DATA / case["filename"]).convert("RGB"))
        want = case["result"]
        ok = got == want
        exact += ok
        distance += edit_distance(got, want)
        characters += len(want)

        elapsed = (time.perf_counter() - started) * 1000
        print(f"{case['filename']}  {'OK  ' if ok else 'DIFF'}  {elapsed:6.0f}ms  {got}")
        if not ok:
            print(f"{'':>26}expected  {want}")

    print(
        f"\nexact match  {exact}/{len(cases)}"
        f"\nchar error   {distance / characters:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
