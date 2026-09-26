import type { RegionShape } from '../detect/types.ts'
import type { Patch } from './balloon.ts'
import type { EraseRegion } from './types.ts'

/**
 * Builds the single-channel mask Telea erases: 255 where text is, 0 elsewhere.
 * A region with a shape (text-map.ts) is erased by that shape, its text core
 * grown by a disk and clipped to the box, which avoids the box corners where a
 * loose box crosses a round bubble's outline. Other regions are erased whole.
 */

/**
 * Pixels to grow each box by. Zero: Telea fills from the ring just outside the
 * mask, and growing it pushes that ring onto the bubble outline, which then
 * smears inward.
 */
const DEFAULT_PADDING = 0

export function buildMask(
  width: number,
  height: number,
  regions: EraseRegion[],
  padding: number = DEFAULT_PADDING,
): Uint8Array {
  const mask = new Uint8Array(width * height)
  for (const region of regions) {
    const patch = regionPatch(width, height, region, padding)
    if (patch) addPatch(mask, width, patch)
  }
  return mask
}

/** Set every pixel of `patch` in a page-sized 0/255 mask. */
export function addPatch(mask: Uint8Array, width: number, patch: Patch): void {
  for (let y = 0; y < patch.height; y++) {
    for (let x = 0; x < patch.width; x++) {
      if (patch.data[y * patch.width + x]) mask[(patch.y + y) * width + patch.x + x] = 255
    }
  }
}

/** One region's erase mask, over its box clipped to the page; null when empty. */
export function regionPatch(
  width: number,
  height: number,
  { box, shape }: EraseRegion,
  padding: number = DEFAULT_PADDING,
): Patch | null {
  const left = Math.max(0, Math.floor(box.x - padding))
  const top = Math.max(0, Math.floor(box.y - padding))
  const right = Math.min(width, Math.ceil(box.x + box.width + padding))
  const bottom = Math.min(height, Math.ceil(box.y + box.height + padding))
  if (right <= left || bottom <= top) return null

  const patch = { x: left, y: top, width: right - left, height: bottom - top, data: new Uint8Array(0) }
  patch.data = new Uint8Array(patch.width * patch.height)
  if (shape) paintShape(patch, shape)
  else patch.data.fill(1)
  return patch
}

/**
 * Where the lettering certainly is, for balloon.ts to sample colour and start
 * its fill: a shape's core cells, or the middle half of a bare box.
 */
export function seedPatch(width: number, height: number, { box, shape }: EraseRegion): Patch | null {
  if (shape) return regionPatch(width, height, { box, shape: { ...shape, grow: 0 } })
  const middle = { x: box.x + box.width / 4, y: box.y + box.height / 4, width: box.width / 2, height: box.height / 2 }
  return regionPatch(width, height, { box: middle })
}

/** Mark every pixel within `shape.grow` of a core cell; a pixel belongs to the cell its top-left corner falls in. */
function paintShape(patch: Patch, shape: RegionShape): void {
  const { x: left, y: top, width: w, height: h } = patch
  const distance = new Float64Array(w * h)
  for (let y = 0; y < h; y++) {
    const row = Math.floor((top + y - shape.y) / shape.cellHeight)
    for (let x = 0; x < w; x++) {
      const column = Math.floor((left + x - shape.x) / shape.cellWidth)
      const core =
        row >= 0 &&
        row < shape.rows &&
        column >= 0 &&
        column < shape.columns &&
        shape.cells[row * shape.columns + column] === 1
      distance[y * w + x] = core ? 0 : Infinity
    }
  }

  squaredDistanceTransform(distance, w, h)
  const limit = shape.grow * shape.grow
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (distance[y * w + x]! <= limit) patch.data[y * w + x] = 1
    }
  }
}

/**
 * Exact squared Euclidean distance to the nearest zero, in place: Felzenszwalb
 * and Huttenlocher's two separable passes over lower envelopes of parabolas.
 * Cells start at 0 (core) or Infinity.
 */
function squaredDistanceTransform(grid: Float64Array, w: number, h: number): void {
  const size = Math.max(w, h)
  const f = new Float64Array(size)
  const d = new Float64Array(size)
  const v = new Int32Array(size)
  const z = new Float64Array(size + 1)

  const pass = (n: number): void => {
    // Parabolas rooted at Infinity never reach the envelope; an all-Infinity line stays so.
    let k = -1
    for (let q = 0; q < n; q++) {
      if (f[q] === Infinity) continue
      if (k < 0) {
        k = 0
        v[0] = q
        z[0] = -Infinity
        z[1] = Infinity
        continue
      }
      let s = (f[q]! + q * q - (f[v[k]!]! + v[k]! * v[k]!)) / (2 * q - 2 * v[k]!)
      while (s <= z[k]!) {
        k--
        s = (f[q]! + q * q - (f[v[k]!]! + v[k]! * v[k]!)) / (2 * q - 2 * v[k]!)
      }
      k++
      v[k] = q
      z[k] = s
      z[k + 1] = Infinity
    }
    if (k < 0) {
      d.fill(Infinity, 0, n)
      return
    }
    k = 0
    for (let q = 0; q < n; q++) {
      while (z[k + 1]! < q) k++
      const r = q - v[k]!
      d[q] = r * r + f[v[k]!]!
    }
  }

  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = grid[y * w + x]!
    pass(h)
    for (let y = 0; y < h; y++) grid[y * w + x] = d[y]!
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = grid[y * w + x]!
    pass(w)
    for (let x = 0; x < w; x++) grid[y * w + x] = d[x]!
  }
}

/** Share of the image the mask covers, which is what makes Telea slow. */
export function maskCoverage(mask: Uint8Array): number {
  let covered = 0
  for (const value of mask) {
    if (value !== 0) covered++
  }
  return mask.length > 0 ? covered / mask.length : 0
}
