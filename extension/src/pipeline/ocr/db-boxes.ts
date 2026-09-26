/**
 * Turns a DB text detector's probability map into line boxes: threshold,
 * connected components, then grow each back by the training shrink (the
 * unclip, closed-form on a rectangle). Equivalent to PaddleOCR's
 * findContours + pyclipper, since only bounding boxes are used.
 */

export interface LineBox {
  x: number
  y: number
  width: number
  height: number
  /** Mean map probability inside the component, PaddleOCR's box_thresh score. */
  score: number
}

/** PaddleOCR's det_db_thresh: map value above which a pixel counts as text. */
const BINARY_THRESHOLD = 0.3

/** PaddleOCR's det_db_box_thresh: mean probability a component must reach. */
const BOX_THRESHOLD = 0.6

/** PaddleOCR's det_db_unclip_ratio. */
const UNCLIP_RATIO = 1.5

/** Smaller than this in either axis is noise, not a line. */
const MIN_SIDE = 3

/**
 * Connected components of `probability > BINARY_THRESHOLD`, scaled by
 * `scaleX`/`scaleY`. Iterative flood fill, since recursion would overflow.
 */
export function linesFromProbabilityMap(
  probability: Float32Array,
  width: number,
  height: number,
  scaleX: number,
  scaleY: number,
): LineBox[] {
  const seen = new Uint8Array(width * height)
  const stack: number[] = []
  const boxes: LineBox[] = []

  for (let start = 0; start < probability.length; start++) {
    if (seen[start] !== 0 || probability[start]! <= BINARY_THRESHOLD) continue

    let left = width
    let top = height
    let right = -1
    let bottom = -1
    let total = 0
    let count = 0

    seen[start] = 1
    stack.push(start)
    while (stack.length > 0) {
      const index = stack.pop()!
      const x = index % width
      const y = (index - x) / width

      if (x < left) left = x
      if (x > right) right = x
      if (y < top) top = y
      if (y > bottom) bottom = y
      total += probability[index]!
      count++

      // 4-connectivity (PaddleOCR uses 8): it cannot leak diagonally into a neighbour.
      if (x > 0 && seen[index - 1] === 0 && probability[index - 1]! > BINARY_THRESHOLD) {
        seen[index - 1] = 1
        stack.push(index - 1)
      }
      if (x + 1 < width && seen[index + 1] === 0 && probability[index + 1]! > BINARY_THRESHOLD) {
        seen[index + 1] = 1
        stack.push(index + 1)
      }
      if (y > 0 && seen[index - width] === 0 && probability[index - width]! > BINARY_THRESHOLD) {
        seen[index - width] = 1
        stack.push(index - width)
      }
      const below = index + width
      if (y + 1 < height && seen[below] === 0 && probability[below]! > BINARY_THRESHOLD) {
        seen[below] = 1
        stack.push(below)
      }
    }

    const boxWidth = right - left + 1
    const boxHeight = bottom - top + 1
    if (boxWidth < MIN_SIDE || boxHeight < MIN_SIDE) continue

    const score = total / count
    if (score < BOX_THRESHOLD) continue

    // Unclip: grow back what the shrunk-label training objective removed.
    const grow = (boxWidth * boxHeight * UNCLIP_RATIO) / (2 * (boxWidth + boxHeight))
    boxes.push({
      x: (left - grow) * scaleX,
      y: (top - grow) * scaleY,
      width: (boxWidth + grow * 2) * scaleX,
      height: (boxHeight + grow * 2) * scaleY,
      score,
    })
  }

  return boxes
}
