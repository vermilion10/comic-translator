import type { LineBox } from './db-boxes.ts'

/**
 * Puts line boxes into reading order. The detector can split one line into
 * several boxes at slightly different heights, so a plain sort interleaves
 * neighbouring lines. Rows are grouped by box centres, not overlap: the DB
 * unclip grows every box, so consecutive lines always overlap.
 */

interface Row {
  centre: number
  /** The thinnest box in the row, which sets how far a centre may stray. */
  thickness: number
  boxes: LineBox[]
}

/**
 * `vertical` swaps the axes: horizontal text is rows top to bottom, boxes left
 * to right; vertical CJK is columns right to left, boxes top to bottom.
 */
export function inReadingOrder(boxes: LineBox[], vertical: boolean): LineBox[] {
  if (boxes.length === 0) return []

  const centre = (box: LineBox) => (vertical ? box.x + box.width / 2 : box.y + box.height / 2)
  const thickness = (box: LineBox) => (vertical ? box.width : box.height)
  const along = (box: LineBox) => (vertical ? box.y : box.x)

  const rows: Row[] = []
  for (const box of [...boxes].sort((a, b) => centre(a) - centre(b))) {
    const existing = rows.find(
      (row) =>
        Math.abs(centre(box) - row.centre) < 0.5 * Math.min(thickness(box), row.thickness),
    )
    if (existing) {
      existing.boxes.push(box)
      existing.thickness = Math.min(existing.thickness, thickness(box))
    } else {
      rows.push({ centre: centre(box), thickness: thickness(box), boxes: [box] })
    }
  }

  if (vertical) rows.reverse()
  return rows.flatMap((row) => [...row.boxes].sort((a, b) => along(a) - along(b)))
}
