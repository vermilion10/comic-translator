/**
 * Decides which regions get a caption plate behind the typeset text, and its
 * colour. A plate is for text on artwork, where there is no bubble to sit in.
 *
 * The signal is how much of the region its two dominant tones cover: lettering
 * on a flat field (a bubble, a black panel, a gutter) is two-tone, artwork is
 * not. The threshold is conservative, firing only where text is clearly on
 * art. It is measured on the source page, since erasing replaces the evidence.
 * The larger tone is also the plate's colour, so both come from one pass.
 */

export interface Rgb {
  r: number
  g: number
  b: number
}

/** One of a region's dominant tones: its mean colour and how much it holds. */
export interface Tone extends Rgb {
  /** Share of the whole region claimed by this tone. */
  share: number
}

export interface Field {
  /** Share of the region its dominant tones account for. */
  coverage: number
  /** The tone holding most of the region: the ground, since lettering is the minority. */
  ground: Rgb
}

/** Pixels of context added on each side before measuring. */
export const PLATE_MARGIN = 6

/** Luma distance at which two pixels still count as the same tone. */
export const TONE_TOLERANCE = 24

const TONE_BINS = 64

/** Two-tone coverage at or above this means there is a field to sit on. */
export const PLATE_TONE_COVERAGE = 0.73

/** What `ground` falls back to when the region has no pixels to sample. */
const NO_GROUND: Rgb = { r: 255, g: 255, b: 255 }

/** ITU-R BT.601 luma, matching PIL's `convert('L')`, which the threshold was measured with. */
function toLuma(rgba: Uint8ClampedArray): Uint8Array {
  const luma = new Uint8Array(rgba.length / 4)
  for (let i = 0, p = 0; p < luma.length; i += 4, p++) {
    luma[p] = (rgba[i]! * 299 + rgba[i + 1]! * 587 + rgba[i + 2]! * 114) / 1000
  }
  return luma
}

/**
 * A region's strongest tones and their shares, greedily: take the tallest
 * histogram bin, claim everything within TONE_TOLERANCE, repeat. Ordered by
 * peak height, not share, so pick the ground by share (`sampleField`).
 */
export function dominantTones(rgba: Uint8ClampedArray, tones = 2): Tone[] {
  const luma = toLuma(rgba)
  if (luma.length === 0) return []

  const claimed = new Uint8Array(luma.length)
  let remaining = luma.length
  const found: Tone[] = []

  for (let round = 0; round < tones && remaining > 0; round++) {
    const histogram = new Uint32Array(TONE_BINS)
    for (let i = 0; i < luma.length; i++) {
      if (claimed[i] === 0) histogram[(luma[i]! * TONE_BINS) >> 8]!++
    }

    let peakBin = 0
    for (let bin = 1; bin < TONE_BINS; bin++) {
      if (histogram[bin]! > histogram[peakBin]!) peakBin = bin
    }
    const peak = ((peakBin + 0.5) * 256) / TONE_BINS

    let count = 0
    let red = 0
    let green = 0
    let blue = 0
    for (let i = 0; i < luma.length; i++) {
      if (claimed[i] !== 0) continue
      if (Math.abs(luma[i]! - peak) > TONE_TOLERANCE) continue
      claimed[i] = 1
      remaining--
      count++
      red += rgba[i * 4]!
      green += rgba[i * 4 + 1]!
      blue += rgba[i * 4 + 2]!
    }
    if (count === 0) break

    found.push({ r: red / count, g: green / count, b: blue / count, share: count / luma.length })
  }

  return found
}

/** Coverage and ground tone of one region, from a single pass over it. */
export function sampleField(rgba: Uint8ClampedArray): Field {
  const tones = dominantTones(rgba)
  if (tones.length === 0) return { coverage: 1, ground: NO_GROUND }

  let coverage = 0
  let ground = tones[0]!
  for (const tone of tones) {
    coverage += tone.share
    if (tone.share > ground.share) ground = tone
  }
  return { coverage, ground: { r: ground.r, g: ground.g, b: ground.b } }
}

/** WCAG relative luminance, which is not the BT.601 luma used above. */
function relativeLuminance({ r, g, b }: Rgb): number {
  const channel = (value: number): number => {
    const c = value / 255
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
  }
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
}

/** WCAG contrast ratio between two relative luminances. */
function contrastRatio(a: number, b: number): number {
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05)
}

/** Relative luminance where black and white ink are equally legible (4.58:1 both). */
const INK_CROSSOVER = 0.1791

/** Plate opacity: semi-opaque so the art underneath still shows. */
export const PLATE_ALPHA = 0.82

/** WCAG AA for body text, the bar the ink has to clear against its plate. */
export const PLATE_MIN_CONTRAST = 4.5

export interface PlateStyle {
  /** '#000000' or '#ffffff', whichever this plate carries better. */
  ink: string
  /** Opacity to fill at, raised to 1 where blending would cost legibility. */
  alpha: number
}

/** One channel of `tone` blended `alpha` of the way over a flat `over`. */
function blend(tone: Rgb, alpha: number, over: number): Rgb {
  const mix = (value: number): number => alpha * value + (1 - alpha) * over
  return { r: mix(tone.r), g: mix(tone.g), b: mix(tone.b) }
}

/**
 * Ink and opacity for a plate of this tone. The better of black and white ink
 * never falls below 4.58:1 against the tone itself. Blending can break that
 * for mid-grey tones, so those plates are drawn opaque.
 */
export function plateStyle(ground: Rgb): PlateStyle {
  const ink = relativeLuminance(ground) > INK_CROSSOVER ? '#000000' : '#ffffff'
  const inkLuminance = ink === '#000000' ? 0 : 1
  // The worst the erase stage could do to this plate is drag it toward the ink.
  const worst = blend(ground, PLATE_ALPHA, inkLuminance * 255)
  const alpha =
    contrastRatio(relativeLuminance(worst), inkLuminance) < PLATE_MIN_CONTRAST ? 1 : PLATE_ALPHA
  return { ink, alpha }
}

/**
 * The field under a region: `box` plus PLATE_MARGIN, clipped to the page. The
 * plate decision and the erase stage's text-on-art decision share this window.
 */
export function fieldAround(
  rgba: Uint8ClampedArray,
  pageWidth: number,
  pageHeight: number,
  box: { x: number; y: number; width: number; height: number },
): Field {
  const x = Math.max(0, Math.floor(box.x) - PLATE_MARGIN)
  const y = Math.max(0, Math.floor(box.y) - PLATE_MARGIN)
  const width = Math.min(pageWidth - x, Math.ceil(box.width) + PLATE_MARGIN * 2)
  const height = Math.min(pageHeight - y, Math.ceil(box.height) + PLATE_MARGIN * 2)
  if (width <= 0 || height <= 0) return { coverage: 1, ground: NO_GROUND }
  const window = new Uint8ClampedArray(width * height * 4)
  for (let row = 0; row < height; row++) {
    const from = ((y + row) * pageWidth + x) * 4
    window.set(rgba.subarray(from, from + width * 4), row * width * 4)
  }
  return sampleField(window)
}

/**
 * Text drawn straight on artwork, by the caption-plate test. It is left
 * unerased and labelled, since erasing it off art leaves a smear.
 */
export function isTextOnArt(field: Field): boolean {
  return field.coverage < PLATE_TONE_COVERAGE
}
