# Comic Translator

A Chrome extension that translates comics, manga, manhua and manhwa in place.
It finds the lettering on a page, reads it, erases it, translates it and
typesets the translation back into the balloons, all inside the browser.

**This is the TextSeg version.** Text detection uses TextSeg, the project's own
detector: a text-segmentation model that predicts a per-pixel text map. The
training and evaluation code is included under [`ml/`](ml/).

## Features

- **Source languages:** Japanese, Chinese and Korean.
- **Target languages:** English, and Indonesian with cloud translation.
- **Detection:** TextSeg finds text regions, including lettering in joined
  balloons and sound effects drawn on the art.
- **Recognition:** manga-ocr for Japanese, including vertical text. PaddleOCR
  for Chinese and Korean.
- **Erasing:**
  - Text in a balloon is erased and filled with the balloon's own colour.
    The outline is left intact, including between joined balloons.
  - Other text is reconstructed with OpenCV's Telea inpainting.
  - Text drawn straight on artwork is left in place and labelled instead.
- **Typesetting:** the translation is laid out inside the balloon, and split
  across the lobes of a joined balloon.
- **Two ways to translate:**
  - *On this device* (default): a local model on WebGPU, with no account,
    key or network after the first download.
  - *Cloud*: Gemini, for devices without WebGPU. It needs your own API key.
- **Works on reader sites:** a floating control finds the page image itself,
  so it also works where a site blocks right-click. You can pick an image by
  hand, draw a box around text the detector missed, and restore the original.
- **Split pages:** pages cut into several image tiles are recognised and
  reported.

## Requirements

- Chrome or another Chromium-based browser with Manifest V3 support.
- WebGPU for on-device translation. Devices without it can use cloud
  translation.
- To build from source: Node.js 20.19 or newer.

## Installation

### From a release

1. Download `comic-translator-<version>.zip` from the
   [Releases](https://github.com/vermilion10/comic-translator/releases) page
   and unzip it.
2. Open `chrome://extensions` and turn on **Developer mode**.
3. Click **Load unpacked** and select the unzipped folder.

### From source

```bash
git clone https://github.com/vermilion10/comic-translator.git
cd comic-translator/extension
npm ci
npm run build
```

`npm run build` first downloads the model weights into `public/models/`.
Each file is pinned to a Hugging Face revision and checked against a SHA-256
digest. Later builds reuse the downloaded files and work offline.

Then load `extension/dist` with **Load unpacked**, as above.

## Usage

1. Open a page with a comic image.
2. Click the **文A** button at the bottom left and press **Translate**. You
   can also right-click the image and choose **Translate this comic page**.
3. The first run downloads the models:
   - The detector and OCR models ship with the extension and load in seconds.
   - The on-device translation model (about 1.5 GB) downloads once and is
     cached.

The floating control also offers:

- **Pick image:** choose a different image on the page.
- **Draw a box:** translate one region the detector missed.
- **Restore:** put the original image back.
- **Settings** (gear icon): open the settings page.

### Settings

| Setting | Description |
|---|---|
| Source language | The script the page is written in. It selects the OCR model, so set it before translating. |
| Translate into | English, or Indonesian (cloud only). |
| Translation | *On this device* or *Cloud (Gemini)*. Cloud needs a Gemini API key. |
| Detection threshold | Higher values translate fewer, more certain regions (default 0.60). |
| Model cache | Shows and clears the cached on-device translation model. |

### Privacy

On-device translation keeps everything on your computer. Cloud translation
sends the recognised text to Google's Gemini API, whose free tier may use it
to improve their models. The API key is stored unencrypted in the browser's
extension storage, on this device only.

## How it works

Each page goes through five stages, run by one shared offscreen document so
the models load once per browser session:

| Stage | Implementation |
|---|---|
| Detect | TextSeg (MobileNetV3 + FPN text map), onnxruntime-web |
| OCR | manga-ocr (Japanese); PP-OCRv6 and korean PP-OCRv5 (Chinese, Korean), onnxruntime-web |
| Erase | Balloon fill, or OpenCV.js Telea inpainting in a sandboxed page |
| Translate | Gemma 2 2B (`gemma-2-2b-jpn-it`) via WebLLM, or Gemini (`gemini-3.5-flash-lite`) |
| Render | Canvas 2D typesetting |

onnxruntime-web uses up to four threads, enabled by making the extension's
own pages cross-origin isolated.

## Repository layout

```
extension/           The Chrome extension (TypeScript, Vite, CRXJS)
  src/pipeline/      Detect, OCR, inpaint, translate and render stages
  src/content/       Page integration and the floating control
  src/ui/            Offscreen pipeline, settings page, sandbox, dev harness
  scripts/           Model download and parity checks
ml/                  TextSeg training and evaluation (Python)
  textseg/           Model, dataset, loss, training, ONNX export
  eval/              Detection and OCR scoring with reference sets
  scripts/           Threshold sweeps, erase-mask measurement, fixtures
  notebooks/         Kaggle notebooks for the page cache and training
  train/             Training labels, hard negatives and ignore regions
```

## Development

From `extension/`:

```bash
npm run dev          # development build with hot reload
npm run build        # type-check and production build
npm run lint         # ESLint
```

The dev harness runs every stage on a single dropped image, with each stage's
output shown. It is at `chrome-extension://<extension-id>/src/ui/harness/index.html`.

### Parity checks

The text-map decoder and the balloon fill exist in both TypeScript and
Python, and the two must agree. To check:

```bash
python ml/scripts/dump_text_map_fixture.py --model extension/public/models/textseg.onnx
node extension/scripts/check-text-map.mjs

python ml/scripts/dump_balloon_fixture.py
node extension/scripts/check-balloon.mjs
```

## Training and evaluation (`ml/`)

Install the Python dependencies (PyTorch CPU is enough for evaluation):

```bash
python -m venv ml/.venv
ml/.venv/bin/pip install torch timm onnx onnxruntime numpy pillow scipy
```

**Evaluate.** Scoring downloads the reference pages into `ml/eval/data/` on
first run. The reference files store page ids and boxes, not images.

```bash
python ml/eval/score_detect.py                 # detection: coverage, IoU, per-slice recall
python ml/eval/score.py                        # Japanese OCR
python ml/eval/score.py --pages                # Japanese OCR on multi-column balloons
python ml/eval/score.py --engine paddle        # Chinese OCR
python ml/eval/score.py --engine paddle --language ko   # Korean OCR
```

**Train.** Training runs on a GPU (the notebooks target Kaggle):

1. `python ml/textseg/pack_kaggle.py` packs the code and labels into one zip.
   Upload it as a Kaggle dataset.
2. `ml/notebooks/textseg_cache.ipynb` (CPU) fetches and caches the training
   pages once.
3. `ml/notebooks/textseg_train.ipynb` (GPU) trains, exports to ONNX and
   scores each seed.

`ml/textseg/train.py --smoke` runs the full training path on a few pages on
CPU. `ml/textseg/export.py` exports a checkpoint to ONNX. The evaluation set
is never trained on: `build_cache.py` refuses any page listed in
`ml/eval/held_out.json`.

## Third-party models

| Model | Use | License |
|---|---|---|
| [vermilion10/manga-textseg](https://huggingface.co/vermilion10/manga-textseg) | Text detection | Project model, timm ImageNet backbone (Apache-2.0) |
| [kha-white/manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base) (ONNX by onnx-community) | Japanese OCR | Apache-2.0 |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) PP-OCRv6 / korean PP-OCRv5 | Chinese and Korean OCR | Apache-2.0 |
| [Gemma 2 2B JPN](https://huggingface.co/google/gemma-2-2b-jpn-it) via [WebLLM](https://github.com/mlc-ai/web-llm) | On-device translation | Gemma Terms of Use |
| [OpenCV.js](https://opencv.org/) | Inpainting | Apache-2.0 |

## License

The project code is released under the [MIT License](LICENSE). The models it
downloads keep their own licenses, listed above.
