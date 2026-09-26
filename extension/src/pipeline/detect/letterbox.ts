/**
 * Resize-with-padding into a square, preserving aspect ratio, as the detector
 * was trained.
 */
export interface Letterbox {
  /** Scale applied to the source image before padding. */
  scale: number
  /** Padding added on the left and top, in target pixels. */
  padX: number
  padY: number
}

/** Neutral grey, the padding colour the detector was trained with. */
const PAD_COLOR = 'rgb(114, 114, 114)'

export function computeLetterbox(
  sourceWidth: number,
  sourceHeight: number,
  target: number,
): Letterbox {
  const scale = Math.min(target / sourceWidth, target / sourceHeight)
  const width = Math.round(sourceWidth * scale)
  const height = Math.round(sourceHeight * scale)

  return {
    scale,
    padX: Math.floor((target - width) / 2),
    padY: Math.floor((target - height) / 2),
  }
}

/** Draw `source` letterboxed into a square canvas of `target` pixels. */
export function drawLetterboxed(
  source: ImageBitmap,
  target: number,
  layout: Letterbox,
): ImageData {
  const canvas = new OffscreenCanvas(target, target)
  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error('could not acquire a 2d context for letterboxing')
  }

  context.fillStyle = PAD_COLOR
  context.fillRect(0, 0, target, target)
  context.drawImage(
    source,
    layout.padX,
    layout.padY,
    Math.round(source.width * layout.scale),
    Math.round(source.height * layout.scale),
  )

  return context.getImageData(0, 0, target, target)
}

/** Map a coordinate from letterboxed space back onto the source image. */
export function undoLetterbox(value: number, pad: number, scale: number): number {
  return (value - pad) / scale
}
