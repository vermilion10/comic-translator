import * as ort from 'onnxruntime-web/wasm'

import { type LineBox, linesFromProbabilityMap } from './db-boxes.ts'

/**
 * PP-OCR's text-line detector: finds the lines or columns inside a region.
 * Mirrored by ml/eval/score.py's PaddleRecognizer._lines.
 */

/** Longest side the detector sees; PaddleOCR's det_limit_side_len. */
const DET_LIMIT = 960

/** The detector's input sides must be multiples of its stride. */
const DET_STRIDE = 32

/** ImageNet mean and standard deviation, in the BGR order the export wants. */
const DET_MEAN = [0.406, 0.456, 0.485]
const DET_STD = [0.225, 0.224, 0.229]

export interface TextLineFinder {
  /** Lines in `region`, in its own pixel coordinates. */
  find(region: OffscreenCanvas): Promise<LineBox[]>
  release(): Promise<void>
}

export async function createTextLineFinder(weights: Uint8Array): Promise<TextLineFinder> {
  const session = await ort.InferenceSession.create(weights, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  const input = session.inputNames[0]
  const output = session.outputNames[0]
  if (!input || !output) throw new Error('the text-line detector exposes no input or output tensor')

  const scratch = new OffscreenCanvas(1, 1)
  const context = scratch.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('could not acquire a 2d context for the text-line detector')

  /** Detector input: long side capped, both sides a multiple of the stride. */
  function shape(width: number, height: number): [number, number] {
    const scale = Math.min(1, DET_LIMIT / Math.max(width, height))
    const snap = (value: number) =>
      Math.max(DET_STRIDE, Math.round((value * scale) / DET_STRIDE) * DET_STRIDE)
    return [snap(width), snap(height)]
  }

  return {
    async find(region: OffscreenCanvas): Promise<LineBox[]> {
      const [width, height] = shape(region.width, region.height)
      scratch.width = width
      scratch.height = height
      context.imageSmoothingQuality = 'high'
      context.drawImage(region, 0, 0, region.width, region.height, 0, 0, width, height)
      const { data } = context.getImageData(0, 0, width, height)

      const pixels = width * height
      const values = new Float32Array(3 * pixels)
      for (let i = 0; i < pixels; i++) {
        // BGR, scaled to 0..1, then ImageNet-normalised.
        values[i] = (data[i * 4 + 2]! / 255 - DET_MEAN[0]!) / DET_STD[0]!
        values[pixels + i] = (data[i * 4 + 1]! / 255 - DET_MEAN[1]!) / DET_STD[1]!
        values[pixels * 2 + i] = (data[i * 4]! / 255 - DET_MEAN[2]!) / DET_STD[2]!
      }

      const outputs = await session.run({
        [input]: new ort.Tensor('float32', values, [1, 3, height, width]),
      })
      const map = outputs[output]
      if (!map) throw new Error(`the text-line detector produced no "${output}" tensor`)

      return linesFromProbabilityMap(
        map.data as Float32Array,
        width,
        height,
        region.width / width,
        region.height / height,
      )
    },

    release: () => session.release(),
  }
}
