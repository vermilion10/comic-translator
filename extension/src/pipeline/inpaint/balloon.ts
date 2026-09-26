import type { Box } from '../detect/types.ts'
import { type Rgb, sampleField } from '../render/plate.ts'

/**
 * Clip an erase region to its balloon and fill it flat with the balloon's
 * colour. Telea fills from the ring just outside the mask, so a mask that
 * crosses a balloon outline smears it; clipped to the interior, the ring *is*
 * the outline, so the region is painted flat instead.
 *
 * Per region, in a window around its mask:
 * 1. Ground colour: plate.ts's sampleField over the seed (text core, or the
 *    middle of the box). Too little two-tone coverage means text on art: fall back.
 * 2. Balloon: pixels within COLOR_TOLERANCE of the ground, the components
 *    reaching the seed, holes filled. Letters are holes; the outline is not.
 * 3. Guard: if more than GUARD of the region's ink is letter-shaped dark
 *    shapes outside the balloon, they would survive as ghost text: fall back.
 * 4. Closed balloons (clear of the window edge, at most INTERIOR_MAX times the
 *    mask) are painted whole, erasing columns the detector missed.
 * 5. Lobes: the largest rectangle in each closed balloon, plus a second when
 *    the remainder is at least SECOND_LOBE of the first (a joined balloon),
 *    for the renderer to set text in.
 *
 * Mirrors ml/eval/balloon.py; scripts/check-balloon.mjs checks they agree.
 */

export const FIELD_COVERAGE = 0.73
export const COLOR_TOLERANCE = 40
const INK_LUMA = 128
export const GUARD = 0.05
const PAD_FRACTION = 0.5
const PAD_MIN = 8
const INTERIOR_MAX = 3
const LOBE_MIN_SHARE = 0.15
const LOBE_INSET = 0.06
const LOBE_INSET_MIN = 2
const LOBE_MIN_SIDE = 8
const SECOND_LOBE = 0.35

/** A page-aligned rectangle of a single-channel mask, 1 where set. */
export interface Patch {
  x: number
  y: number
  width: number
  height: number
  data: Uint8Array
}

export interface BalloonFill {
  /** Where to paint, in page coordinates. */
  patch: Patch
  color: Rgb
  /** Rectangles inside the region's closed balloons, for the renderer; empty when none closed. */
  lobes: Box[]
}

/** Same truncated BT.601 luma as plate.ts. */
function lumaAt(rgba: Uint8ClampedArray, index: number): number {
  return Math.floor((rgba[index * 4]! * 299 + rgba[index * 4 + 1]! * 587 + rgba[index * 4 + 2]! * 114) / 1000)
}

/** 4-connected components of the set cells; 0 is unset, labels from 1. */
function label4(set: Uint8Array, w: number, h: number): Int32Array {
  const labels = new Int32Array(w * h)
  const stack = new Int32Array(w * h)
  let next = 0
  for (let start = 0; start < set.length; start++) {
    if (!set[start] || labels[start]) continue
    next++
    let depth = 0
    stack[depth++] = start
    labels[start] = next
    const visit = (n: number): void => {
      if (set[n] && !labels[n]) {
        labels[n] = next
        stack[depth++] = n
      }
    }
    while (depth > 0) {
      const cell = stack[--depth]!
      const y = Math.floor(cell / w)
      const x = cell - y * w
      if (y > 0) visit(cell - w)
      if (y < h - 1) visit(cell + w)
      if (x > 0) visit(cell - 1)
      if (x < w - 1) visit(cell + 1)
    }
  }
  return labels
}

/** The set cells plus every unset cell not 4-connected to the grid's edge. */
function fillHoles(set: Uint8Array, w: number, h: number): Uint8Array {
  const unset = new Uint8Array(w * h)
  for (let i = 0; i < w * h; i++) unset[i] = set[i] ? 0 : 1
  const background = label4(unset, w, h)
  const edge = new Set<number>()
  for (let x = 0; x < w; x++) {
    edge.add(background[x]!)
    edge.add(background[(h - 1) * w + x]!)
  }
  for (let y = 0; y < h; y++) {
    edge.add(background[y * w]!)
    edge.add(background[y * w + w - 1]!)
  }
  const filled = new Uint8Array(w * h)
  for (let i = 0; i < w * h; i++) filled[i] = set[i] || !edge.has(background[i]!) ? 1 : 0
  return filled
}

/**
 * The largest all-set axis-aligned rectangle, [x, y, w, h], first found on a
 * tie (histogram of run heights, stack of starts), matching the Python side.
 */
function largestRectangle(set: Uint8Array, w: number, h: number): [number, number, number, number] {
  const heights = new Int32Array(w)
  let best: [number, number, number, number, number] = [0, 0, 0, 0, 0]
  const starts: number[] = []
  const stackHeights: number[] = []
  for (let row = 0; row < h; row++) {
    for (let column = 0; column < w; column++) {
      heights[column] = set[row * w + column] ? heights[column]! + 1 : 0
    }
    starts.length = 0
    stackHeights.length = 0
    for (let column = 0; column <= w; column++) {
      const current = column < w ? heights[column]! : 0
      let start = column
      while (stackHeights.length > 0 && stackHeights[stackHeights.length - 1]! >= current) {
        const begin = starts.pop()!
        const height = stackHeights.pop()!
        const area = height * (column - begin)
        if (area > best[0]) best = [area, begin, row - height + 1, column - begin, height]
        start = begin
      }
      starts.push(start)
      stackHeights.push(current)
    }
  }
  return [best[1], best[2], best[3], best[4]]
}

/**
 * Where to erase and with what, or null when the region should go to Telea
 * unchanged. `rgba` is the whole page; both patches are this region's.
 */
export function clipToBalloon(
  rgba: Uint8ClampedArray,
  pageWidth: number,
  pageHeight: number,
  base: Patch,
  seed: Patch,
): BalloonFill | null {
  // Bounds of the set pixels, not the patch, so the window matches Python's.
  let top = Infinity
  let bottom = -Infinity
  let left = Infinity
  let right = -Infinity
  for (let y = 0; y < base.height; y++) {
    for (let x = 0; x < base.width; x++) {
      if (!base.data[y * base.width + x]) continue
      top = Math.min(top, base.y + y)
      bottom = Math.max(bottom, base.y + y + 1)
      left = Math.min(left, base.x + x)
      right = Math.max(right, base.x + x + 1)
    }
  }
  if (top === Infinity) return null

  const pad = Math.floor(Math.max(PAD_MIN, PAD_FRACTION * Math.max(bottom - top, right - left)))
  const y0 = Math.max(0, top - pad)
  const y1 = Math.min(pageHeight, bottom + pad)
  const x0 = Math.max(0, left - pad)
  const x1 = Math.min(pageWidth, right + pad)
  const w = x1 - x0
  const h = y1 - y0

  const at = (patch: Patch, px: number, py: number): boolean => {
    const x = px - patch.x
    const y = py - patch.y
    return x >= 0 && y >= 0 && x < patch.width && y < patch.height && patch.data[y * patch.width + x] === 1
  }
  const inBase = new Uint8Array(w * h)
  const inSeed = new Uint8Array(w * h)
  let seeded = 0
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const b = at(base, x0 + x, y0 + y)
      inBase[y * w + x] = b ? 1 : 0
      if (b && at(seed, x0 + x, y0 + y)) {
        inSeed[y * w + x] = 1
        seeded++
      }
    }
  }

  // The ground, sampled over the seed (or the whole region), in row-major order.
  const sampleFrom = seeded > 0 ? inSeed : inBase
  let sampleCount = 0
  for (const v of sampleFrom) sampleCount += v
  const sample = new Uint8ClampedArray(sampleCount * 4)
  let k = 0
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!sampleFrom[y * w + x]) continue
      const i = ((y0 + y) * pageWidth + x0 + x) * 4
      sample[k++] = rgba[i]!
      sample[k++] = rgba[i + 1]!
      sample[k++] = rgba[i + 2]!
      sample[k++] = 255
    }
  }
  if (sampleCount === 0) return null
  const { coverage, ground } = sampleField(sample)
  if (coverage < FIELD_COVERAGE) return null

  const field = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = ((y0 + y) * pageWidth + x0 + x) * 4
      field[y * w + x] =
        Math.abs(rgba[i]! - ground.r) <= COLOR_TOLERANCE &&
        Math.abs(rgba[i + 1]! - ground.g) <= COLOR_TOLERANCE &&
        Math.abs(rgba[i + 2]! - ground.b) <= COLOR_TOLERANCE
          ? 1
          : 0
    }
  }
  const components = label4(field, w, h)
  const reached = new Set<number>()
  for (let i = 0; i < w * h; i++) {
    if (inSeed[i] && field[i]) reached.add(components[i]!)
  }
  if (reached.size === 0) return null
  // Ascending, like np.unique, so the order matches Python's.
  const reachedLabels = [...reached].sort((a, b) => a - b)

  // The balloon, then its holes: unset cells not connected to the window edge.
  const reachedCells = new Uint8Array(w * h)
  for (let i = 0; i < w * h; i++) reachedCells[i] = reached.has(components[i]!) ? 1 : 0
  const balloon = fillHoles(reachedCells, w, h)

  const ink = new Uint8Array(w * h)
  let inkTotal = 0
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      ink[i] = lumaAt(rgba, (y0 + y) * pageWidth + x0 + x) < INK_LUMA ? 1 : 0
      if (ink[i] && inBase[i]) inkTotal++
    }
  }
  if (inkTotal > 0) {
    const shapes = label4(ink, w, h)
    const leaving = new Set<number>()
    for (let i = 0; i < w * h; i++) {
      if (ink[i] && !inBase[i]) leaving.add(shapes[i]!)
    }
    let lost = 0
    for (let i = 0; i < w * h; i++) {
      if (ink[i] && inBase[i] && !leaving.has(shapes[i]!) && !balloon[i]) lost++
    }
    if (lost / inkTotal > GUARD) return null
  }

  const data = new Uint8Array(w * h)
  let baseCount = 0
  for (let i = 0; i < w * h; i++) {
    data[i] = inBase[i] && balloon[i] ? 1 : 0
    baseCount += inBase[i]!
  }

  // Closed balloons are painted whole.
  const closed: { cells: Uint8Array; area: number }[] = []
  for (const label of reachedLabels) {
    const own = new Uint8Array(w * h)
    for (let i = 0; i < w * h; i++) own[i] = components[i] === label ? 1 : 0
    const part = fillHoles(own, w, h)
    let area = 0
    let touches = false
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (!part[y * w + x]) continue
        area++
        if (y === 0 || y === h - 1 || x === 0 || x === w - 1) touches = true
      }
    }
    if (!touches && area <= INTERIOR_MAX * baseCount) {
      closed.push({ cells: part, area })
      for (let i = 0; i < w * h; i++) if (part[i]) data[i] = 1
    }
  }

  const lobes: Box[] = []
  const closedArea = closed.reduce((sum, part) => sum + part.area, 0)
  for (const { cells, area } of closed) {
    if (area < LOBE_MIN_SHARE * closedArea) continue
    let top = h
    let bottom = -1
    let left = w
    let right = -1
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (!cells[y * w + x]) continue
        top = Math.min(top, y)
        bottom = Math.max(bottom, y)
        left = Math.min(left, x)
        right = Math.max(right, x)
      }
    }
    const cw = right - left + 1
    const ch = bottom - top + 1
    const crop = new Uint8Array(cw * ch)
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) crop[y * cw + x] = cells[(top + y) * w + left + x]!
    }
    const first = largestRectangle(crop, cw, ch)
    const rectangles = [first]
    const [fx, fy, fw, fh] = first
    for (let y = fy; y < fy + fh; y++) crop.fill(0, y * cw + fx, y * cw + fx + fw)
    const second = largestRectangle(crop, cw, ch)
    if (second[2] * second[3] >= SECOND_LOBE * fw * fh) rectangles.push(second)
    for (const [rx, ry, rw, rh] of rectangles) {
      const inset = Math.max(LOBE_INSET_MIN, LOBE_INSET * Math.min(rw, rh))
      if (Math.min(rw, rh) - 2 * inset < LOBE_MIN_SIDE) continue
      lobes.push({
        x: x0 + left + rx + inset,
        y: y0 + top + ry + inset,
        width: rw - 2 * inset,
        height: rh - 2 * inset,
      })
    }
  }

  return {
    patch: { x: x0, y: y0, width: w, height: h, data },
    color: { r: Math.floor(ground.r + 0.5), g: Math.floor(ground.g + 0.5), b: Math.floor(ground.b + 0.5) },
    lobes,
  }
}
