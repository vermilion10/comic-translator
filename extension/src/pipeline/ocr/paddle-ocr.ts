import * as ort from 'onnxruntime-web/wasm'

import type { Box } from '../detect/types.ts'
import { configureOrtThreads } from '../ort-threads.ts'
import { createTextLineFinder } from './text-lines.ts'
import { inReadingOrder } from './reading-order.ts'
import type { OcrResult, Recognition, RecognitionListener, Recognizer } from './types.ts'

/**
 * OCR stage for Chinese and Korean, backed by PaddleOCR (Apache-2.0) on
 * onnxruntime-web. One detector serves both; the recogniser and character set
 * differ (PP-OCRv6 small for Chinese, korean_PP-OCRv5_mobile for Korean).
 *   det  in  "x"             float32 [1, 3, H, W]   H, W multiples of 32
 *        out "fetch_name_0"  float32 [1, 1, H, W]   per-pixel text probability
 *   rec  in  "x"             float32 [1, 3, 48, W]
 *        out "fetch_name_0"  float32 [1, T, C]      CTC probabilities
 *                                                   C = 18710 zh, 11947 ko
 *
 * The recogniser reads exactly one line, so the detector first cuts each
 * region into lines. It reads only horizontally: a vertical column is cut into
 * glyph cells at ink gaps and laid out left to right, which works far worse
 * than horizontal text. Vertical Korean is untested.
 */

/** The recogniser is shape-locked to this height; the width is free. */
const REC_HEIGHT = 48

/** PaddleOCR pads recogniser input up to a multiple of this many pixels. */
const REC_WIDTH_STEP = 8

/** Nothing narrower than this is worth a recogniser pass. */
const MIN_REC_WIDTH = 160

/** A line box taller than this many times its width is a vertical column. */
const VERTICAL_ASPECT = 1.5

/** A row of a column counts as ink if at least this fraction of it is dark. */
const INK_ACTIVE_FRACTION = 0.03

/** A gap must last this fraction of the column's width to count, so a glyph's own hollow does not split it. */
const INK_GAP_FRACTION = 0.045

/** Cells shorter than this fraction of the column's width are noise, not glyphs. */
const MIN_CELL_FRACTION = 0.3

export interface PaddleOcrOptions {
  detUrl: string
  recUrl: string
  /** One character per line, in the order the export's class indices use. */
  dictUrl: string
  /** Joins two lines of one region: a space for Korean, which spaces its words; nothing otherwise. */
  lineSeparator?: string
}

async function fetchBytes(url: string, what: string): Promise<Uint8Array> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`could not load ${what} from ${url}: ${response.status.toString()}`)
  }
  return new Uint8Array(await response.arrayBuffer())
}

function context2d(canvas: OffscreenCanvas, what: string): OffscreenCanvasRenderingContext2D {
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error(`could not acquire a 2d context for ${what}`)
  return context
}

export async function createPaddleOcr(options: PaddleOcrOptions): Promise<Recognizer> {
  configureOrtThreads()

  const [detWeights, recWeights, dictResponse] = await Promise.all([
    fetchBytes(options.detUrl, 'text detector weights'),
    fetchBytes(options.recUrl, 'text recogniser weights'),
    fetch(options.dictUrl),
  ])
  if (!dictResponse.ok) {
    throw new Error(
      `could not load the OCR dictionary from ${options.dictUrl}: ${dictResponse.status.toString()}`,
    )
  }

  // Class 0 is the CTC blank and the last class a space, so the export has two
  // more classes than the file has lines.
  const entries = (await dictResponse.text()).split('\n')
  if (entries.at(-1) === '') entries.pop()
  const charset = ['', ...entries, ' ']

  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  }
  const [lineFinder, recogniser] = await Promise.all([
    createTextLineFinder(detWeights),
    ort.InferenceSession.create(recWeights, sessionOptions),
  ])

  const recInput = recogniser.inputNames[0]
  const recOutput = recogniser.outputNames[0]
  if (!recInput || !recOutput) {
    throw new Error('a PP-OCR model exposes no input or output tensor')
  }

  const lineSeparator = options.lineSeparator ?? ''

  const region = new OffscreenCanvas(1, 1)
  const regionContext = context2d(region, 'the OCR region')
  const scratch = new OffscreenCanvas(1, 1)
  const scratchContext = context2d(scratch, 'the OCR line')

  /**
   * Per-row ink count within a column of `region`. Ink may be dark or light,
   * so the split comes from Otsu's method on the column's own histogram.
   */
  function inkRowCounts(x: number, y: number, width: number, height: number): Float64Array {
    const { data } = regionContext.getImageData(x, y, width, height)
    const luma = new Uint8ClampedArray(width * height)
    const histogram = new Uint32Array(256)
    for (let i = 0; i < luma.length; i++) {
      const value = Math.round(
        0.299 * data[i * 4]! + 0.587 * data[i * 4 + 1]! + 0.114 * data[i * 4 + 2]!,
      )
      luma[i] = value
      histogram[value]!++
    }

    let sumAll = 0
    for (let v = 0; v < 256; v++) sumAll += v * histogram[v]!
    let sumBelow = 0
    let weightBelow = 0
    let bestThreshold = 0
    let bestVariance = -1
    for (let v = 0; v < 256; v++) {
      weightBelow += histogram[v]!
      if (weightBelow === 0) continue
      const weightAbove = luma.length - weightBelow
      if (weightAbove === 0) break
      sumBelow += v * histogram[v]!
      const meanBelow = sumBelow / weightBelow
      const meanAbove = (sumAll - sumBelow) / weightAbove
      const variance = weightBelow * weightAbove * (meanBelow - meanAbove) ** 2
      if (variance > bestVariance) {
        bestVariance = variance
        bestThreshold = v
      }
    }

    let darkCount = 0
    for (let i = 0; i < luma.length; i++) if (luma[i]! <= bestThreshold) darkCount++
    const darkIsInk = darkCount < luma.length / 2

    const counts = new Float64Array(height)
    for (let row = 0; row < height; row++) {
      let count = 0
      for (let col = 0; col < width; col++) {
        if (luma[row * width + col]! <= bestThreshold === darkIsInk) count++
      }
      counts[row] = count
    }
    return counts
  }

  /** Row ranges to cut a column into, at real ink gaps; the whole column when there are none. */
  function inkSegments(counts: Float64Array, width: number): [number, number][] {
    const activeThreshold = Math.max(1, width * INK_ACTIVE_FRACTION)
    const minGap = Math.max(1, Math.round(width * INK_GAP_FRACTION))
    const minCell = width * MIN_CELL_FRACTION

    const raw: [number, number][] = []
    let start = -1
    let gap = 0
    for (let row = 0; row < counts.length; row++) {
      if (counts[row]! > activeThreshold) {
        if (start === -1) start = row
        gap = 0
      } else if (start !== -1) {
        gap++
        if (gap >= minGap) {
          raw.push([start, row - gap + 1])
          start = -1
          gap = 0
        }
      }
    }
    if (start !== -1) raw.push([start, counts.length])

    const segments = raw.filter(([top, bottom]) => bottom - top >= minCell)
    return segments.length > 0 ? segments : [[0, counts.length]]
  }

  /** Lay a vertical column of glyphs out left to right for the horizontal-only recogniser. */
  function unrollColumn(x: number, y: number, width: number, height: number): void {
    const counts = inkRowCounts(x, y, width, height)
    const segments = inkSegments(counts, width)
    scratch.width = width * segments.length
    scratch.height = width
    scratchContext.imageSmoothingQuality = 'high'
    scratchContext.fillStyle = '#ffffff'
    scratchContext.fillRect(0, 0, scratch.width, scratch.height)
    segments.forEach(([top, bottom], cell) => {
      scratchContext.drawImage(
        region,
        x,
        y + top,
        width,
        bottom - top,
        cell * width,
        0,
        width,
        width,
      )
    })
  }

  /** Read whatever is currently in `scratch` as one line of text. */
  async function readScratch(): Promise<{ text: string; probability: number }> {
    const aspect = scratch.width / Math.max(1, scratch.height)
    const padded = Math.max(
      MIN_REC_WIDTH,
      Math.ceil((REC_HEIGHT * aspect) / REC_WIDTH_STEP) * REC_WIDTH_STEP,
    )
    const drawn = Math.max(1, Math.min(padded, Math.round(REC_HEIGHT * aspect)))

    const line = new OffscreenCanvas(padded, REC_HEIGHT)
    const lineContext = context2d(line, 'the OCR line strip')
    lineContext.imageSmoothingQuality = 'high'
    lineContext.drawImage(scratch, 0, 0, scratch.width, scratch.height, 0, 0, drawn, REC_HEIGHT)
    const { data } = lineContext.getImageData(0, 0, padded, REC_HEIGHT)

    const pixels = padded * REC_HEIGHT
    const values = new Float32Array(3 * pixels)
    for (let y = 0; y < REC_HEIGHT; y++) {
      for (let x = 0; x < drawn; x++) {
        const source = (y * padded + x) * 4
        const target = y * padded + x
        // BGR, mapped to -1..1; the right padding stays 0, as PaddleOCR pads.
        values[target] = data[source + 2]! / 127.5 - 1
        values[pixels + target] = data[source + 1]! / 127.5 - 1
        values[pixels * 2 + target] = data[source]! / 127.5 - 1
      }
    }

    const outputs = await recogniser.run({
      [recInput!]: new ort.Tensor('float32', values, [1, 3, REC_HEIGHT, padded]),
    })
    const logits = outputs[recOutput!]
    if (!logits) throw new Error(`the text recogniser produced no "${recOutput!}" tensor`)

    const steps = logits.dims[1] ?? 0
    const classes = logits.dims[2] ?? charset.length
    const probabilities = logits.data as Float32Array

    // CTC greedy: best class per step, dropping blanks and repeats.
    let text = ''
    let total = 0
    let kept = 0
    let previous = -1
    for (let step = 0; step < steps; step++) {
      const offset = step * classes
      let best = 0
      for (let index = 1; index < classes; index++) {
        if (probabilities[offset + index]! > probabilities[offset + best]!) best = index
      }
      if (best !== previous && best !== 0) {
        text += charset[best] ?? ''
        total += probabilities[offset + best]!
        kept++
      }
      previous = best
    }
    return { text, probability: kept > 0 ? total / kept : 0 }
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
        const startedAt = performance.now()

        region.width = Math.max(1, Math.round(box.width))
        region.height = Math.max(1, Math.round(box.height))
        regionContext.imageSmoothingQuality = 'high'
        regionContext.drawImage(
          source,
          box.x,
          box.y,
          box.width,
          box.height,
          0,
          0,
          region.width,
          region.height,
        )

        const preprocessedAt = performance.now()
        const lines = await lineFinder.find(region)
        const detectedAt = performance.now()

        const tall = lines.filter((line) => line.height > line.width * VERTICAL_ASPECT).length
        const vertical = lines.length > 0 && tall * 2 >= lines.length
        const ordered = inReadingOrder(lines, vertical)

        let text = ''
        let total = 0
        let read = 0

        if (lines.length === 0) {
          // No line found: read the region as one; the map can miss short single lines.
          scratch.width = region.width
          scratch.height = region.height
          scratchContext.drawImage(region, 0, 0)
          const single = await readScratch()
          text = single.text
          total = single.probability
          read = single.text === '' ? 0 : 1
        }

        for (const line of ordered) {
          const x = Math.max(0, Math.round(line.x))
          const y = Math.max(0, Math.round(line.y))
          const width = Math.min(region.width - x, Math.round(line.width))
          const height = Math.min(region.height - y, Math.round(line.height))
          if (width < 4 || height < 4) continue

          if (line.height > line.width * VERTICAL_ASPECT) {
            unrollColumn(x, y, width, height)
          } else {
            scratch.width = width
            scratch.height = height
            scratchContext.drawImage(region, x, y, width, height, 0, 0, width, height)
          }

          const { text: lineText, probability } = await readScratch()
          if (lineText === '') continue
          text += text === '' ? lineText : lineSeparator + lineText
          total += probability
          read++
        }

        const recognition: Recognition = {
          box,
          text,
          // Recognition.confidence is a mean log-probability; CTC gives a mean probability.
          confidence: read > 0 ? Math.log(Math.max(Number.MIN_VALUE, total / read)) : 0,
        }
        recognitions.push(recognition)
        onRecognized?.(recognition, index)

        timings.preprocess += preprocessedAt - startedAt
        timings.encode += detectedAt - preprocessedAt
        timings.decode += performance.now() - detectedAt
      }

      return { recognitions, timings }
    },

    async dispose(): Promise<void> {
      await Promise.all([lineFinder.release(), recogniser.release()])
    },
  }
}
