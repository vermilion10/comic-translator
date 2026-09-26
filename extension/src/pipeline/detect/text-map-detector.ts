import * as ort from 'onnxruntime-web/wasm'

import { configureOrtThreads } from '../ort-threads.ts'
import { computeLetterbox, drawLetterboxed } from './letterbox.ts'
import {
  DEFAULT_TEXT_MAP_DECODE,
  decodeTextMap,
  OUTPUT_STRIDE,
  type TextMapDecodeOptions,
} from './text-map.ts'
import type { DetectResult, Detector } from './types.ts'

/**
 * Detect stage backed by textseg, a MobileNetV3 + FPN text-probability model.
 *   input  float32 [1, 3, 1280, 1280]  RGB, CHW, 0..1, letterboxed
 *   output float32 [1, 1, 320, 320]    text probability, one cell per 4 pixels
 * text-map.ts turns the map into boxes.
 */

const INPUT_SIZE = 1280

export interface TextMapDetectorOptions {
  /** URL of the .onnx weights, e.g. chrome.runtime.getURL('models/…'). */
  modelUrl: string
  /** Minimum region score to report. */
  scoreThreshold?: number
  decode?: TextMapDecodeOptions
}

export async function createTextMapDetector(options: TextMapDetectorOptions): Promise<Detector> {
  const threshold = options.scoreThreshold ?? 0.25
  const decode = options.decode ?? DEFAULT_TEXT_MAP_DECODE

  configureOrtThreads()

  const response = await fetch(options.modelUrl)
  if (!response.ok) {
    throw new Error(`could not load detector weights from ${options.modelUrl}: ${response.status}`)
  }
  const session = await ort.InferenceSession.create(new Uint8Array(await response.arrayBuffer()), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })

  const inputName = session.inputNames[0]
  const outputName = session.outputNames[0]
  if (!inputName || !outputName) {
    throw new Error('detector model exposes no input or output tensor')
  }

  return {
    async detect(source: ImageBitmap): Promise<DetectResult> {
      const startedAt = performance.now()

      const layout = computeLetterbox(source.width, source.height, INPUT_SIZE)
      const { data: rgba } = drawLetterboxed(source, INPUT_SIZE, layout)
      const pixels = INPUT_SIZE * INPUT_SIZE
      const chw = new Float32Array(3 * pixels)
      for (let i = 0; i < pixels; i++) {
        chw[i] = rgba[i * 4]! / 255
        chw[pixels + i] = rgba[i * 4 + 1]! / 255
        chw[pixels * 2 + i] = rgba[i * 4 + 2]! / 255
      }

      const preprocessedAt = performance.now()

      const outputs = await session.run({
        [inputName]: new ort.Tensor('float32', chw, [1, 3, INPUT_SIZE, INPUT_SIZE]),
      })
      const output = outputs[outputName]
      if (!output) {
        throw new Error(`detector produced no "${outputName}" tensor`)
      }
      const mapSize = output.dims[3] ?? 0
      if (mapSize * OUTPUT_STRIDE !== INPUT_SIZE) {
        throw new Error(`unexpected text map shape [${output.dims.join(', ')}]`)
      }

      const inferredAt = performance.now()

      const detections = decodeTextMap(
        output.data as Float32Array,
        mapSize,
        layout,
        source.width,
        source.height,
        threshold,
        decode,
      )

      return {
        detections,
        timings: {
          preprocess: preprocessedAt - startedAt,
          inference: inferredAt - preprocessedAt,
          decode: performance.now() - inferredAt,
        },
      }
    },

    async dispose(): Promise<void> {
      await session.release()
    },
  }
}
