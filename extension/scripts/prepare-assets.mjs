/**
 * Downloads the model weights into public/models/ before a build, each pinned
 * to a revision and checked against a digest; a file already present is kept.
 *
 * OpenCV.js is copied from node_modules and loaded as a classic script, not
 * bundled: it is CommonJS whose export is a Promise, and the bundler's interop
 * breaks awaiting it.
 */
import { createHash } from 'node:crypto'
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')

// Pinned revisions, so the weights cannot change between builds.
const ASSETS = [
  // The text detector (ml/textseg/).
  {
    repo: 'vermilion10/manga-textseg',
    revision: 'cf8d6beffbb2a77de901736d6c5dea889c117f97',
    path: 'textseg.onnx',
    dest: 'public/models/textseg.onnx',
    sha256: '7591878efe3e1167c1cc8ba4b619a78deb41c6ed7cd91adce3b4ca2f970a1915',
  },
  // manga-ocr, int8: the best of the quantizations on the reference set; fp32 is 343 MB.
  {
    repo: 'onnx-community/manga-ocr-base-ONNX',
    revision: 'f9023406bb2f6b17df67bc4a327c56ecd20611f0',
    path: 'onnx/encoder_model_quantized.onnx',
    dest: 'public/models/manga-ocr-encoder.onnx',
    sha256: 'ddd1af56963093795705fa38da6ce7e6567d1658e7c7359db7e13fcd37dbf279',
  },
  {
    repo: 'onnx-community/manga-ocr-base-ONNX',
    revision: 'f9023406bb2f6b17df67bc4a327c56ecd20611f0',
    path: 'onnx/decoder_model_quantized.onnx',
    dest: 'public/models/manga-ocr-decoder.onnx',
    sha256: '2e7177d2b0a59f1c612b694ed70c13971bee765cc2b2bc7bc9376e4753652f27',
  },
  // The ONNX export ships no tokenizer, so the vocabulary comes from the upstream repo.
  {
    repo: 'kha-white/manga-ocr-base',
    revision: 'aa6573bd10b0d446cbf622e29c3e084914df9741',
    path: 'vocab.txt',
    dest: 'public/models/manga-ocr-vocab.txt',
    sha256: '344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d',
  },
  // PP-OCRv6 small, for Chinese: PaddlePaddle's own ONNX exports.
  {
    repo: 'PaddlePaddle/PP-OCRv6_small_det_onnx',
    revision: '28fe5895c24fd108c19eb3e8479f4ab385fbfc62',
    path: 'inference.onnx',
    dest: 'public/models/ppocr-det.onnx',
    sha256: 'd73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e',
  },
  {
    repo: 'PaddlePaddle/PP-OCRv6_small_rec_onnx',
    revision: 'b8f84f0b80c529de40b4fbb3544b84fa7233a513',
    path: 'inference.onnx',
    dest: 'public/models/ppocr-rec.onnx',
    sha256: '5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634',
  },
  // The recogniser's character set, a YAML list flattened below.
  {
    repo: 'PaddlePaddle/PP-OCRv6_small_rec_onnx',
    revision: 'b8f84f0b80c529de40b4fbb3544b84fa7233a513',
    path: 'inference.yml',
    dest: 'public/models/ppocr-rec.yml',
    sha256: 'ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1',
  },
  // Korean shares the detector; v5 is the newest Korean recogniser PaddlePaddle publishes.
  {
    repo: 'PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx',
    revision: '5c6f574b8e2230adf4287b33e736d71b9fabd28e',
    path: 'inference.onnx',
    dest: 'public/models/ppocr-rec-ko.onnx',
    sha256: '92f0b7785e64fc9090106a241cf4c1eb97472824558272751b88a2a4476d3a08',
  },
  {
    repo: 'PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx',
    revision: '5c6f574b8e2230adf4287b33e736d71b9fabd28e',
    path: 'inference.yml',
    dest: 'public/models/ppocr-rec-ko.yml',
    sha256: 'f757fa1c40e99edcf27e9cce879b93eb2a51fa46f5ef39095689b8c37dd75998',
  },
]

/** Files built from a pinned, checksummed download. */
const DERIVED = [
  {
    from: 'public/models/ppocr-rec.yml',
    dest: 'public/models/ppocr-dict.txt',
    build: extractCharacterDict,
  },
  {
    from: 'public/models/ppocr-rec-ko.yml',
    dest: 'public/models/ppocr-dict-ko.txt',
    build: extractCharacterDict,
  },
]

/**
 * Writes PostProcess.character_dict from a PaddleOCR inference.yml as one
 * character per line, so no YAML parser ships in the extension. Punctuation
 * entries are single-quoted, where '' is a literal quote.
 */
function extractCharacterDict(yaml) {
  const lines = yaml.split(NEWLINE)
  const start = lines.findIndex((line) => line.startsWith('  character_dict:'))
  if (start === -1) throw new Error('no character_dict in the recogniser config')

  const characters = []
  for (const line of lines.slice(start + 1)) {
    if (!line.startsWith('  - ')) {
      if (line.trim() === '') continue
      break
    }
    const raw = line.slice(4)
    characters.push(
      raw.length >= 2 && raw.startsWith("'") && raw.endsWith("'")
        ? raw.slice(1, -1).replaceAll("''", "'")
        : raw,
    )
  }
  if (characters.length === 0) throw new Error('the recogniser config has an empty character_dict')
  return `${characters.join(NEWLINE)}${NEWLINE}`
}

const NEWLINE = String.fromCharCode(10)

/** Copied, not downloaded: package-lock.json pins the version. */
const VENDOR = [
  {
    from: 'node_modules/@techstark/opencv-js/dist/opencv.js',
    dest: 'public/vendor/opencv.js',
  },
]

function sha256(buffer) {
  return createHash('sha256').update(buffer).digest('hex')
}

async function readIfPresent(path) {
  try {
    return await readFile(path)
  } catch (error) {
    if (error.code === 'ENOENT') return null
    throw error
  }
}

async function ensureAsset(asset) {
  const dest = join(root, asset.dest)

  const existing = await readIfPresent(dest)
  if (existing && sha256(existing) === asset.sha256) {
    console.log(`✓ ${asset.dest} (cached)`)
    return
  }
  if (existing) {
    console.log(`… ${asset.dest} checksum mismatch, re-downloading`)
  }

  const url = `https://huggingface.co/${asset.repo}/resolve/${asset.revision}/${asset.path}`
  console.log(`↓ ${url}`)

  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`download failed: ${response.status} ${response.statusText}`)
  }
  const buffer = Buffer.from(await response.arrayBuffer())

  const actual = sha256(buffer)
  if (actual !== asset.sha256) {
    throw new Error(
      `checksum mismatch for ${asset.path}\n  expected ${asset.sha256}\n  actual   ${actual}`,
    )
  }

  await mkdir(dirname(dest), { recursive: true })
  await writeFile(dest, buffer)
  console.log(`✓ ${asset.dest} (${(buffer.byteLength / 1e6).toFixed(1)} MB)`)
}

async function ensureDerived(entry) {
  const source = join(root, entry.from)
  const dest = join(root, entry.dest)

  const built = entry.build(await readFile(source, 'utf8'))
  const existing = await readIfPresent(dest)
  if (existing && existing.toString('utf8') === built) {
    console.log(`✓ ${entry.dest} (cached)`)
    return
  }

  await mkdir(dirname(dest), { recursive: true })
  await writeFile(dest, built, 'utf8')
  const lines = built.split(NEWLINE).length - 1
  console.log(`✓ ${entry.dest} (${lines} characters, derived)`)
}

async function ensureVendored(entry) {
  const source = join(root, entry.from)
  const dest = join(root, entry.dest)

  const [current, existing] = await Promise.all([
    readFile(source).catch(() => null),
    readIfPresent(dest),
  ])
  if (!current) {
    throw new Error(`${entry.from} is missing; run npm install first`)
  }
  if (existing && sha256(existing) === sha256(current)) {
    console.log(`✓ ${entry.dest} (cached)`)
    return
  }

  await mkdir(dirname(dest), { recursive: true })
  await copyFile(source, dest)
  console.log(`✓ ${entry.dest} (${(current.byteLength / 1e6).toFixed(1)} MB, copied)`)
}

// Sequential, so a failure is easy to read.
for (const asset of ASSETS) {
  await ensureAsset(asset)
}
for (const entry of VENDOR) {
  await ensureVendored(entry)
}
for (const entry of DERIVED) {
  await ensureDerived(entry)
}
