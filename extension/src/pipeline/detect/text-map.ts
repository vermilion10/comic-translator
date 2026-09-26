import { type Letterbox, undoLetterbox } from './letterbox.ts'
import type { Detection, RegionShape } from './types.ts'

/**
 * textseg's probability map -> scored boxes, the DBNet way. A mirror of
 * ml/textseg/detector.py's decode; change one and change the other. No
 * onnxruntime import, so scripts/check-text-map.mjs can run it under Node.
 */

/** Input pixels per map cell. */
export const OUTPUT_STRIDE = 4

/** The shrink the training labels were painted with; must match ml/textseg/dataset.py. */
const SHRINK_RATIO = 0.4

/** Components smaller than this many map cells are noise. */
const MIN_CELLS = 3

export interface TextMapDecodeOptions {
  /** A cell belongs to a region when its probability is above this. */
  binarize: number
  /** Multiplier on the offset that undoes the label shrink; below 1 gives tighter boxes. */
  unshrinkScale: number
}

/**
 * How far a region's erase shape reaches past its core, as a fraction of the
 * offset that rebuilds its box. The grown core keeps the box's reach along the
 * text and drops its corners, where a loose box crosses a round bubble. At 0.75
 * it erases as much lettering as the whole box with a third of the other ink
 * (ml/scripts/measure_erase_mask.py).
 */
export const ERASE_GROW = 0.75

/** The default decoder. */
export const DEFAULT_TEXT_MAP_DECODE: TextMapDecodeOptions = { binarize: 0.3, unshrinkScale: 1 }

/**
 * The offset D the training shrink took off a (w+2D, h+2D) box: solves
 * D = (w+2D)(h+2D)(1-r^2) / (2(w+h+4D)) with the same 30 fixed-point steps as Python.
 */
export function unshrinkOffset(width: number, height: number, ratio = SHRINK_RATIO): number {
  const k = 1 - ratio ** 2
  let d = 0
  for (let i = 0; i < 30; i++) {
    d = ((width + 2 * d) * (height + 2 * d) * k) / (2 * (width + height + 4 * d))
  }
  return d
}

/**
 * Probability map -> scored boxes in source-image pixels: one region per
 * 4-connected component above `binarize`, scored by its mean probability, its
 * bounding box grown by the unshrink offset. Components never overlap.
 */
export function decodeTextMap(
  prob: Float32Array,
  mapSize: number,
  layout: Letterbox,
  sourceWidth: number,
  sourceHeight: number,
  threshold: number,
  options: TextMapDecodeOptions = DEFAULT_TEXT_MAP_DECODE,
): Detection[] {
  const cells = mapSize * mapSize
  const seen = new Uint8Array(cells)
  const stack = new Int32Array(cells)
  /** The current component's cells, in the order the fill reached them. */
  const members = new Int32Array(cells)
  const detections: Detection[] = []

  /** Queue a neighbour if it is text and not yet claimed; returns the new stack depth. */
  const push = (cell: number, depth: number): number => {
    if (seen[cell] || !(prob[cell]! > options.binarize)) return depth
    seen[cell] = 1
    stack[depth] = cell
    return depth + 1
  }

  for (let start = 0; start < cells; start++) {
    if (seen[start] || !(prob[start]! > options.binarize)) continue

    // Flood fill one component, tracking its size, probability sum and bounds.
    let depth = 0
    stack[depth++] = start
    seen[start] = 1
    let count = 0
    let sum = 0
    let minRow = mapSize
    let maxRow = -1
    let minCol = mapSize
    let maxCol = -1
    while (depth > 0) {
      const cell = stack[--depth]!
      const row = Math.floor(cell / mapSize)
      const col = cell - row * mapSize
      members[count] = cell
      count++
      sum += prob[cell]!
      if (row < minRow) minRow = row
      if (row > maxRow) maxRow = row
      if (col < minCol) minCol = col
      if (col > maxCol) maxCol = col
      if (row > 0) depth = push(cell - mapSize, depth)
      if (row < mapSize - 1) depth = push(cell + mapSize, depth)
      if (col > 0) depth = push(cell - 1, depth)
      if (col < mapSize - 1) depth = push(cell + 1, depth)
    }

    const score = sum / count
    if (score < threshold || count < MIN_CELLS) continue

    let x0 = minCol * OUTPUT_STRIDE
    let x1 = (maxCol + 1) * OUTPUT_STRIDE
    let y0 = minRow * OUTPUT_STRIDE
    let y1 = (maxRow + 1) * OUTPUT_STRIDE
    const offset = unshrinkOffset(x1 - x0, y1 - y0)
    const d = options.unshrinkScale * offset
    x0 -= d
    y0 -= d
    x1 += d
    y1 += d

    const left = Math.max(0, Math.min(undoLetterbox(x0, layout.padX, layout.scale), sourceWidth))
    const top = Math.max(0, Math.min(undoLetterbox(y0, layout.padY, layout.scale), sourceHeight))
    const right = Math.max(0, Math.min(undoLetterbox(x1, layout.padX, layout.scale), sourceWidth))
    const bottom = Math.max(0, Math.min(undoLetterbox(y1, layout.padY, layout.scale), sourceHeight))
    if (right > left && bottom > top) {
      detections.push({
        score,
        box: { x: left, y: top, width: right - left, height: bottom - top },
        shape: componentShape(members, count, mapSize, minRow, maxRow, minCol, maxCol, layout, offset),
      })
    }
  }

  return detections.sort((a, b) => b.score - a.score)
}

/** The component's cells as a RegionShape, grown from the unscaled offset so unshrinkScale only affects the box. */
function componentShape(
  members: Int32Array,
  count: number,
  mapSize: number,
  minRow: number,
  maxRow: number,
  minCol: number,
  maxCol: number,
  layout: Letterbox,
  offset: number,
): RegionShape {
  const columns = maxCol - minCol + 1
  const rows = maxRow - minRow + 1
  const grid = new Uint8Array(columns * rows)
  for (let i = 0; i < count; i++) {
    const cell = members[i]!
    const row = Math.floor(cell / mapSize)
    grid[(row - minRow) * columns + (cell - row * mapSize - minCol)] = 1
  }
  const cellSize = OUTPUT_STRIDE / layout.scale
  return {
    cells: grid,
    columns,
    rows,
    x: undoLetterbox(minCol * OUTPUT_STRIDE, layout.padX, layout.scale),
    y: undoLetterbox(minRow * OUTPUT_STRIDE, layout.padY, layout.scale),
    cellWidth: cellSize,
    cellHeight: cellSize,
    grow: (ERASE_GROW * offset) / layout.scale,
  }
}
