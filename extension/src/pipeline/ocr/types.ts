import type { Box } from '../detect/types.ts'

/** The source text read out of one detected region. */
export interface Recognition {
  box: Box
  /** Post-processed text. Empty when the model produced nothing usable. */
  text: string
  /**
   * Mean per-token log-probability of the decoded text (<= 0). Clean reads sit
   * near 0; junk regions drop to about -0.5 or below.
   */
  confidence: number
}

export interface OcrResult {
  recognitions: Recognition[]
  /** Wall-clock milliseconds, summed over every region. */
  timings: {
    preprocess: number
    encode: number
    decode: number
  }
}

/** Called as each region finishes, so slow multi-region runs can show progress. */
export type RecognitionListener = (recognition: Recognition, index: number) => void

/** The OCR stage: image plus regions (source-image pixels) in, text out. */
export interface Recognizer {
  recognize(
    source: ImageBitmap,
    boxes: Box[],
    onRecognized?: RecognitionListener,
  ): Promise<OcrResult>
  dispose(): Promise<void>
}
