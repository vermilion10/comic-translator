/**
 * The OpenCV.js half of the inpaint stage, run inside the sandboxed page.
 * `cv` is passed in so the 12.7 MB module stays out of other bundles.
 */

/** Minimal surface of OpenCV.js that this module uses. */
export interface OpenCvMatLike {
  data: Uint8Array
  delete(): void
}

export interface OpenCvLike {
  CV_8UC1: number
  CV_8UC4: number
  COLOR_RGBA2RGB: number
  COLOR_RGB2RGBA: number
  INPAINT_TELEA: number
  /** Only the empty form: the four-argument form takes a fill colour, not pixels. */
  Mat: { new (): OpenCvMatLike }
  matFromArray(
    rows: number,
    cols: number,
    type: number,
    data: ArrayLike<number>,
  ): OpenCvMatLike
  cvtColor(src: OpenCvMatLike, dst: OpenCvMatLike, code: number): void
  inpaint(
    src: OpenCvMatLike,
    mask: OpenCvMatLike,
    dst: OpenCvMatLike,
    radius: number,
    flags: number,
  ): void
}

/** How far Telea looks outside the mask for fill colour; OpenCV's default. */
export const DEFAULT_RADIUS = 3

/**
 * Erase the masked regions and reconstruct what was under them. Returns a new
 * buffer; alpha is dropped for OpenCV and restored after.
 */
export function inpaintRgba(
  cv: OpenCvLike,
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  mask: Uint8Array,
  radius: number = DEFAULT_RADIUS,
): Uint8ClampedArray {
  const source = cv.matFromArray(height, width, cv.CV_8UC4, rgba)
  const rgb = new cv.Mat()
  const maskMat = cv.matFromArray(height, width, cv.CV_8UC1, mask)
  const painted = new cv.Mat()
  const result = new cv.Mat()

  try {
    cv.cvtColor(source, rgb, cv.COLOR_RGBA2RGB)
    cv.inpaint(rgb, maskMat, painted, radius, cv.INPAINT_TELEA)
    cv.cvtColor(painted, result, cv.COLOR_RGB2RGBA)
    // Copy before freeing: result.data is a view onto the wasm heap.
    return new Uint8ClampedArray(result.data.slice(0, width * height * 4))
  } finally {
    source.delete()
    rgb.delete()
    maskMat.delete()
    painted.delete()
    result.delete()
  }
}
