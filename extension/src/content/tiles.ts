/**
 * Detects a page that is one image cut into several img elements, as webtoon
 * sites do. The cuts ignore the artwork and can split a bubble, so the control
 * reports the split instead of silently translating one tile. Stitching is not
 * done: the other tiles are usually still lazy-loading placeholders.
 */

/** How far two edges may differ and still count as aligned, in CSS pixels. */
const EDGE_TOLERANCE = 2

/** How many tiles before a stack counts as a split page; two could be a layout. */
const MIN_TILES = 3

export interface Strip {
  /** The tiles, in reading order down the page. */
  tiles: HTMLImageElement[]
  /** Where the queried image sits in that order, counting from 0. */
  index: number
  /** Height of the whole strip in CSS pixels, seams included. */
  height: number
}

interface Placed {
  image: HTMLImageElement
  left: number
  top: number
  width: number
  height: number
}

function place(image: HTMLImageElement): Placed {
  const rect = image.getBoundingClientRect()
  return {
    image,
    left: rect.left + window.scrollX,
    top: rect.top + window.scrollY,
    width: rect.width,
    height: rect.height,
  }
}

/**
 * The run of equally wide, edge-aligned, touching images that `image` belongs
 * to, or null. The touching seam is what separates it from a thumbnail column.
 */
export function findStrip(image: HTMLImageElement, images: HTMLImageElement[]): Strip | null {
  const target = place(image)
  if (target.width <= 0 || target.height <= 0) return null

  const column = images
    .map(place)
    .filter(
      (candidate) =>
        Math.abs(candidate.left - target.left) <= EDGE_TOLERANCE &&
        Math.abs(candidate.width - target.width) <= EDGE_TOLERANCE &&
        candidate.height > 0,
    )
    .sort((a, b) => a.top - b.top)

  const start = column.findIndex((candidate) => candidate.image === image)
  if (start === -1) return null

  const run = [column[start]!]
  for (let k = start; k > 0; k--) {
    const above = column[k - 1]!
    if (Math.abs(column[k]!.top - (above.top + above.height)) > EDGE_TOLERANCE) break
    run.unshift(above)
  }
  for (let k = start; k < column.length - 1; k++) {
    const below = column[k + 1]!
    if (Math.abs(below.top - (column[k]!.top + column[k]!.height)) > EDGE_TOLERANCE) break
    run.push(below)
  }

  if (run.length < MIN_TILES) return null
  const last = run[run.length - 1]!
  return {
    tiles: run.map((placed) => placed.image),
    index: run.findIndex((placed) => placed.image === image),
    height: last.top + last.height - run[0]!.top,
  }
}

/** Which tiles of a strip a page-coordinate rectangle touches, since a drawn box can cross a seam. */
export function tilesTouched(
  strip: Strip,
  drawn: { left: number; top: number; width: number; height: number },
): number[] {
  const touched: number[] = []
  strip.tiles.forEach((tile, index) => {
    const placed = place(tile)
    const overlapWidth =
      Math.min(placed.left + placed.width, drawn.left + drawn.width) -
      Math.max(placed.left, drawn.left)
    const overlapHeight =
      Math.min(placed.top + placed.height, drawn.top + drawn.height) - Math.max(placed.top, drawn.top)
    if (overlapWidth > 0 && overlapHeight > 0) touched.push(index)
  })
  return touched
}
