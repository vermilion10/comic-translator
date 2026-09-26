/** Axis-aligned rectangle in source-image pixel coordinates. */
export interface Box {
  x: number
  y: number
  width: number
  height: number
}

/**
 * Where inside its box a region's text is, so erasing can skip the rest. Cell
 * (c, r) covers x + c * cellWidth to x + (c + 1) * cellWidth, and likewise down.
 */
export interface RegionShape {
  /** Row-major, `columns * rows` long, 1 where the cell is text core. */
  cells: Uint8Array
  columns: number
  rows: number
  x: number
  y: number
  cellWidth: number
  cellHeight: number
  /** How far past the core the text reaches, in source pixels. */
  grow: number
}

/** One detected text region. */
export interface Detection {
  box: Box
  /** Model confidence in [0, 1]. */
  score: number
  /** Only from detectors that can say more than a rectangle; see text-map.ts. */
  shape?: RegionShape
}

export interface DetectResult {
  detections: Detection[]
  /** Wall-clock milliseconds. */
  timings: {
    preprocess: number
    inference: number
    decode: number
  }
}

/** The detect stage: an image in, scored boxes in source-image pixels out. */
export interface Detector {
  detect(source: ImageBitmap): Promise<DetectResult>
  dispose(): Promise<void>
}
