"""Zip the code and label JSON a Kaggle run needs into one upload.

Usage:
    python ml/textseg/pack_kaggle.py        # -> ml/textseg/kaggle/comic-translator-ml.zip

Upload the zip as a private Kaggle Dataset. It holds post ids, boxes and code,
never pages.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml" / "textseg" / "kaggle" / "comic-translator-ml.zip"

INCLUDE = [
    "ml/textseg/*.py",
    "ml/scripts/*.py",
    "ml/eval/*.py",
    "ml/eval/*.json",
    "ml/train/*.json",
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = sorted({p for pattern in INCLUDE for p in ROOT.glob(pattern) if p.is_file()})
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"{OUT} {OUT.stat().st_size / 1e6:.1f} MB, {len(files)} files")


if __name__ == "__main__":
    main()
