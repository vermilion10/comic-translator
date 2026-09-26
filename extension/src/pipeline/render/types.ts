import type { Box } from '../detect/types.ts'
import type { Rgb } from './plate.ts'

/** One block of typeset lines and the rectangle it was fitted to. */
export interface TextBlock {
  fontSize: number
  lines: string[]
  bounds: Box
  overflowed: boolean
}

/** What the render stage is asked to typeset for one region. */
export interface RenderInput {
  box: Box
  text: string
  /**
   * Rectangles inside the region's balloon, in reading order. When present the
   * text is set in them instead of `box`, split across several.
   */
  lobes?: Box[]
  /** Text on art left unerased: set in a small label beside the original. */
  label?: boolean
}

/** One typeset region: the text, and where it actually ended up. */
export interface RenderedRegion {
  /** The region the detect stage found, unchanged. */
  box: Box
  text: string
  fontSize: number
  lines: string[]
  /** Where the text was laid out: `box`, or larger when it had to spill. */
  bounds: Box
  /** True when the text did not fit `box` at the minimum readable size. */
  overflowed: boolean
  /** Every block drawn: one, or one per lobe of a joined balloon. */
  blocks: TextBlock[]
  /** A caption plate was drawn behind this region. See plate.ts. */
  plated: boolean
  /** Set as a label beside the unerased original rather than in place. */
  labelled: boolean
  /** Share of the region in its two dominant tones, which decides `plated`. */
  toneCoverage: number
  /** The plate's fill tone; null when plates are off. */
  plateTone: Rgb | null
}

export interface RenderResult {
  image: ImageBitmap
  regions: RenderedRegion[]
  timings: {
    layout: number
    draw: number
  }
}

/**
 * The render stage: draws the translated text onto the erased image,
 * horizontally, left to right.
 */
export interface Renderer {
  /** `source` is the page before erasing; the plate decision reads what was behind the text. */
  render(
    erased: ImageBitmap,
    source: ImageBitmap,
    regions: RenderInput[],
  ): Promise<RenderResult>
  dispose(): Promise<void>
}
