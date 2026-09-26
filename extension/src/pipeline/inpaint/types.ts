import type { Box, RegionShape } from '../detect/types.ts'

/** One region to erase: a Detection, or a bare `{ box }`. With a shape, only the shape is erased. */
export interface EraseRegion {
  box: Box
  shape?: RegionShape
}

export interface InpaintOptions {
  /**
   * Clip each region to its balloon and paint it with the balloon's colour
   * (balloon.ts). Regions without a flat balloon go to Telea.
   */
  balloonFill?: boolean
  /** Leave text drawn on artwork unerased; the renderer labels it instead. */
  keepTextOnArt?: boolean
}

export interface InpaintResult {
  /** The source image with the given regions erased and reconstructed. */
  image: ImageBitmap
  timings: {
    mask: number
    inpaint: number
  }
  /** How many regions took the balloon fill rather than Telea. */
  filled: number
  /**
   * Per input region: layout rectangles of the balloons it sits in, empty when
   * none. In page order, not reading order.
   */
  lobes: Box[][]
  /** Per input region: left unerased as text on art, to be labelled. */
  onArt: boolean[]
}

/** The inpaint stage: erases regions (source-image pixels) and reconstructs what was behind them. */
export interface Inpainter {
  inpaint(source: ImageBitmap, regions: EraseRegion[], options?: InpaintOptions): Promise<InpaintResult>
  dispose(): Promise<void>
}

/** Messages to and from the sandboxed OpenCV.js page (window.postMessage, transferable buffers). */
export type SandboxRequest = {
  type: 'inpaint-request'
  id: number
  width: number
  height: number
  /** RGBA, 4 bytes per pixel, transferred. */
  rgba: ArrayBuffer
  /** One byte per pixel, 255 where the region should be erased. Transferred. */
  mask: ArrayBuffer
  radius: number
}

export type SandboxResponse =
  | { type: 'inpaint-ready' }
  | { type: 'inpaint-result'; id: number; rgba: ArrayBuffer }
  | { type: 'inpaint-error'; id: number; message: string }
