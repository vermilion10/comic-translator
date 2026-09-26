import * as ort from 'onnxruntime-web/wasm'

import type { Box } from '../detect/types.ts'
import { configureOrtThreads } from '../ort-threads.ts'
import { postProcess } from './postprocess.ts'
import { createTextLineFinder } from './text-lines.ts'
import type { OcrResult, Recognition, RecognitionListener, Recognizer } from './types.ts'

/**
 * OCR stage backed by manga-ocr (kha-white/manga-ocr-base, Apache-2.0): a ViT
 * encoder and a 2-layer BERT decoder, on onnxruntime-web.
 *   encoder  in  "pixel_values"           float32 [1, 3, 224, 224]
 *            out "last_hidden_state"      float32 [1, 197, 768]
 *   decoder  in  "input_ids"              int64   [1, seq]
 *            in  "encoder_hidden_states"  float32 [1, 197, 768]
 *            out "logits"                 float32 [1, seq, 6144]
 * The decoder has no KV cache, so each step re-runs the prefix; fine for a
 * 2-layer decoder and short lines. Decoding is greedy.
 */

/** The ViT export is shape-locked to this resolution; it is not a tunable. */
const INPUT_SIZE = 224

/** From the model's generation_config.json. */
const START_TOKEN = 2
const EOS_TOKEN = 3
const MAX_LENGTH = 300

/** Ids 0..4 are [PAD], [UNK], [CLS], [SEP], [MASK]. */
const FIRST_TEXT_TOKEN = 5

/** Stops a decoder stuck repeating one character on a textless crop. */
const MAX_REPEATS = 16

/**
 * Most characters one read is given. The model squashes its input to 224x224,
 * so a many-column region leaves each character a few pixels; such a region is
 * read in groups of consecutive columns instead (ml/eval/score.py mirrors this).
 */
const CHUNK_CHARS = 18

/** A line box taller than this many times its width is a vertical column. */
const VERTICAL_ASPECT = 1.5

/** Chunk only when this share of the lines found are vertical columns. */
const VERTICAL_SHARE = 0.6

export interface MangaOcrOptions {
  encoderUrl: string
  decoderUrl: string
  vocabUrl: string
  /** PP-OCR's text-line detector, for reading big regions in chunks. */
  lineDetectorUrl: string
}

async function fetchBytes(url: string, what: string): Promise<Uint8Array> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`could not load ${what} from ${url}: ${response.status}`)
  }
  return new Uint8Array(await response.arrayBuffer())
}

export async function createMangaOcr(options: MangaOcrOptions): Promise<Recognizer> {
  configureOrtThreads()

  const [encoderWeights, decoderWeights, lineWeights, vocabResponse] = await Promise.all([
    fetchBytes(options.encoderUrl, 'OCR encoder weights'),
    fetchBytes(options.decoderUrl, 'OCR decoder weights'),
    fetchBytes(options.lineDetectorUrl, 'text-line detector weights'),
    fetch(options.vocabUrl),
  ])

  if (!vocabResponse.ok) {
    throw new Error(
      `could not load OCR vocabulary from ${options.vocabUrl}: ${vocabResponse.status}`,
    )
  }
  // One token per line, and the file ends with a newline.
  const vocab = (await vocabResponse.text()).split('\n')
  if (vocab.at(-1) === '') vocab.pop()

  // 'basic', not 'all': the extra fusions add about 0.6 s of session building
  // and no speed. ml/eval/score.py uses the same level.
  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'basic',
  }
  const [encoder, decoder, lineFinder] = await Promise.all([
    ort.InferenceSession.create(encoderWeights, sessionOptions),
    ort.InferenceSession.create(decoderWeights, sessionOptions),
    createTextLineFinder(lineWeights),
  ])
  const region = new OffscreenCanvas(1, 1)
  const regionContext = region.getContext('2d')
  if (!regionContext) throw new Error('could not acquire a 2d context for the OCR region')

  const encoderOutput = encoder.outputNames[0]
  const logitsName = decoder.outputNames[0]
  if (!encoderOutput || !logitsName) {
    throw new Error('OCR model exposes no output tensor')
  }

  /**
   * Crop one region the way the reference preprocessor does: greyscale,
   * squashed to a square, scaled to 0..1, normalised with mean and std 0.5.
   */
  function toPixelValues(source: ImageBitmap, box: Box): Float32Array {
    const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE)
    const context = canvas.getContext('2d')
    if (!context) {
      throw new Error('could not acquire a 2d context for the OCR crop')
    }

    context.imageSmoothingQuality = 'high'
    context.drawImage(
      source,
      box.x,
      box.y,
      box.width,
      box.height,
      0,
      0,
      INPUT_SIZE,
      INPUT_SIZE,
    )
    const { data } = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE)

    // PIL's BT.601 luma, not the canvas' Rec. 709 greyscale.
    const pixels = INPUT_SIZE * INPUT_SIZE
    const values = new Float32Array(3 * pixels)
    for (let i = 0; i < pixels; i++) {
      const luma =
        0.299 * data[i * 4]! + 0.587 * data[i * 4 + 1]! + 0.114 * data[i * 4 + 2]!
      // (luma / 255 - 0.5) / 0.5, folded into one operation.
      const normalised = luma / 127.5 - 1
      values[i] = normalised
      values[pixels + i] = normalised
      values[pixels * 2 + i] = normalised
    }
    return values
  }

  function detokenize(ids: number[]): string {
    let text = ''
    for (const id of ids) {
      const token = vocab[id]
      if (token === undefined || id < FIRST_TEXT_TOKEN || token.startsWith('<unused')) {
        continue
      }
      text += token
    }
    return postProcess(text)
  }

  /** One manga-ocr pass over one box: the decoded text and its log-probability. */
  async function readBox(
    source: ImageBitmap,
    box: Box,
    timings: OcrResult['timings'],
  ): Promise<{ text: string; logProb: number; steps: number }> {
    const startedAt = performance.now()
    const pixelValues = new ort.Tensor('float32', toPixelValues(source, box), [
      1,
      3,
      INPUT_SIZE,
      INPUT_SIZE,
    ])

    const preprocessedAt = performance.now()
    const encoded = await encoder.run({ pixel_values: pixelValues })
    const hidden = encoded[encoderOutput!]
    if (!hidden) {
      throw new Error(`OCR encoder produced no "${encoderOutput!}" tensor`)
    }

    const encodedAt = performance.now()

    const ids: number[] = [START_TOKEN]
    let logProb = 0
    let steps = 0
    let repeats = 0

    for (let step = 0; step < MAX_LENGTH; step++) {
      const outputs = await decoder.run({
        input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]),
        encoder_hidden_states: hidden,
      })
      const logits = outputs[logitsName!]
      if (!logits) {
        throw new Error(`OCR decoder produced no "${logitsName!}" tensor`)
      }

      // Only the distribution for the position after the prefix matters.
      const vocabSize = logits.dims[2] ?? vocab.length
      const row = (logits.data as Float32Array).subarray(
        (ids.length - 1) * vocabSize,
        ids.length * vocabSize,
      )

      let best = 0
      for (let token = 1; token < row.length; token++) {
        if (row[token]! > row[best]!) best = token
      }

      // log softmax at the chosen token, shifted so exp() cannot overflow.
      let sum = 0
      for (const value of row) sum += Math.exp(value - row[best]!)
      logProb += -Math.log(sum)
      steps++

      if (best === EOS_TOKEN) break

      repeats = best === ids.at(-1) ? repeats + 1 : 0
      if (repeats >= MAX_REPEATS) break

      ids.push(best)
    }

    timings.preprocess += preprocessedAt - startedAt
    timings.encode += encodedAt - preprocessedAt
    timings.decode += performance.now() - encodedAt
    return { text: detokenize(ids.slice(1)), logProb, steps }
  }

  /**
   * The boxes to read `box` as: itself, or groups of consecutive vertical
   * columns (right to left) of up to CHUNK_CHARS characters, estimated as
   * height over width since the glyphs are about square.
   */
  async function chunksOf(source: ImageBitmap, box: Box): Promise<Box[]> {
    region.width = Math.max(1, Math.round(box.width))
    region.height = Math.max(1, Math.round(box.height))
    regionContext!.imageSmoothingQuality = 'high'
    regionContext!.drawImage(source, box.x, box.y, box.width, box.height, 0, 0, region.width, region.height)
    const lines = await lineFinder.find(region)
    const vertical = lines.filter((line) => line.height > VERTICAL_ASPECT * line.width)
    if (lines.length < 2 || vertical.length < VERTICAL_SHARE * lines.length) return [box]

    const columns = [...lines].sort((a, b) => b.x + b.width / 2 - (a.x + a.width / 2))
    const groups: (typeof lines)[] = []
    let current: typeof lines = []
    let chars = 0
    for (const column of columns) {
      const estimate = column.height / Math.max(1, column.width)
      if (current.length > 0 && chars + estimate > CHUNK_CHARS) {
        groups.push(current)
        current = []
        chars = 0
      }
      current.push(column)
      chars += estimate
    }
    groups.push(current)
    if (groups.length === 1) return [box]

    // Back to page pixels: the region was drawn at the box's own scale.
    const sx = box.width / region.width
    const sy = box.height / region.height
    return groups.map((group) => {
      const x0 = Math.floor(Math.min(...group.map((c) => c.x)))
      const y0 = Math.floor(Math.min(...group.map((c) => c.y)))
      const x1 = Math.floor(Math.max(...group.map((c) => c.x + c.width)))
      const y1 = Math.floor(Math.max(...group.map((c) => c.y + c.height)))
      return { x: box.x + x0 * sx, y: box.y + y0 * sy, width: (x1 - x0) * sx, height: (y1 - y0) * sy }
    })
  }

  return {
    async recognize(
      source: ImageBitmap,
      boxes: Box[],
      onRecognized?: RecognitionListener,
    ): Promise<OcrResult> {
      const recognitions: Recognition[] = []
      const timings = { preprocess: 0, encode: 0, decode: 0 }

      for (const [index, box] of boxes.entries()) {
        let text = ''
        let logProb = 0
        let steps = 0
        for (const chunk of await chunksOf(source, box)) {
          const read = await readBox(source, chunk, timings)
          text += read.text
          logProb += read.logProb
          steps += read.steps
        }

        const recognition: Recognition = {
          box,
          text,
          confidence: steps > 0 ? logProb / steps : 0,
        }
        recognitions.push(recognition)
        onRecognized?.(recognition, index)
      }

      return { recognitions, timings }
    },

    async dispose(): Promise<void> {
      await Promise.all([encoder.release(), decoder.release(), lineFinder.release()])
    },
  }
}
