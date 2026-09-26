import type { Box } from '../detect/types.ts'

/**
 * Text fitting for the render stage, pure so it runs without a browser.
 * Detector boxes bound vertical source text, so they are often narrower than
 * one English word; `fitText` may widen rather than only shrink.
 */

/** Smallest size still worth reading on a comic page. */
export const MIN_FONT_SIZE = 10

/** Larger than this looks wrong against the surrounding art. */
export const MAX_FONT_SIZE = 28

export const LINE_HEIGHT = 1.15

/**
 * How far the layout may widen past a box too narrow for the text: beside a
 * text column is usually the rest of the bubble, but eventually it is art.
 */
export const MAX_WIDTH_EXPANSION = 2.5

/** Measures a string's width at a given font size, in pixels. */
export type Measure = (text: string, fontSize: number) => number

export interface Fit {
  fontSize: number
  lines: string[]
  bounds: Box
  overflowed: boolean
}

/** Greedy wrap. A word longer than maxWidth gets its own line and overhangs. */
export function wrapText(
  text: string,
  maxWidth: number,
  fontSize: number,
  measure: Measure,
): string[] {
  const words = text.split(/\s+/u).filter((word) => word.length > 0)
  if (words.length === 0) return []

  const lines: string[] = []
  let line = ''
  for (const word of words) {
    const candidate = line === '' ? word : `${line} ${word}`
    if (line !== '' && measure(candidate, fontSize) > maxWidth) {
      lines.push(line)
      line = word
    } else {
      line = candidate
    }
  }
  if (line !== '') lines.push(line)
  return lines
}

function widestLine(lines: string[], fontSize: number, measure: Measure): number {
  return lines.reduce((widest, line) => Math.max(widest, measure(line, fontSize)), 0)
}

function blockHeight(lineCount: number, fontSize: number): number {
  return lineCount * fontSize * LINE_HEIGHT
}

/**
 * The largest font size at which `text` fits `box`. When nothing fits at
 * MIN_FONT_SIZE the text is widened and centred and reported as overflowed,
 * never truncated: the translation is the only copy the reader gets.
 */
export function fitText(text: string, box: Box, measure: Measure): Fit {
  const trimmed = text.trim()
  if (trimmed === '') {
    return { fontSize: MIN_FONT_SIZE, lines: [], bounds: box, overflowed: false }
  }

  // Never start larger than the box could hold on a single line.
  const start = Math.min(MAX_FONT_SIZE, Math.max(MIN_FONT_SIZE, box.height))
  for (let fontSize = start; fontSize >= MIN_FONT_SIZE; fontSize--) {
    const lines = wrapText(trimmed, box.width, fontSize, measure)
    if (
      widestLine(lines, fontSize, measure) <= box.width &&
      blockHeight(lines.length, fontSize) <= box.height
    ) {
      return { fontSize, lines, bounds: box, overflowed: false }
    }
  }

  // Too narrow at the minimum size: widen for the longest word, up to the cap.
  const fontSize = MIN_FONT_SIZE
  const longestWord = trimmed
    .split(/\s+/u)
    .reduce((widest, word) => Math.max(widest, measure(word, fontSize)), 0)
  const width = Math.min(
    Math.max(box.width, longestWord),
    box.width * MAX_WIDTH_EXPANSION,
  )
  const lines = wrapText(trimmed, width, fontSize, measure)
  const height = blockHeight(lines.length, fontSize)

  return {
    fontSize,
    lines,
    bounds: {
      x: box.x + (box.width - width) / 2,
      y: box.y + (box.height - height) / 2,
      width,
      height,
    },
    overflowed: true,
  }
}

/** Ends a sentence; the best place to split, right after one. */
const SENTENCE_END = /[.!?\u2026\u300d\u300f"'\u3002\uff01\uff1f]$/u

/** Ends a clause; the next best. */
const CLAUSE_END = /[,;:)]$/u

/** How far a sentence end, then a clause end, may pull a split point, as a share of the text. */
const SENTENCE_REACH = 0.15
const CLAUSE_REACH = 0.25

/**
 * Divide a translation between the lobes of a joined balloon, one part per
 * weight (lobe area), in order. Each cut goes at the word nearest its share,
 * moved to a nearby sentence or clause end. Never splits a word.
 */
export function splitText(text: string, weights: number[]): string[] {
  const words = text.trim().split(/\s+/u).filter((word) => word.length > 0)
  if (weights.length <= 1 || words.length <= 1) return words.length ? [words.join(' ')] : []

  // Character offset of the end of each word, counting one space between.
  const ends: number[] = []
  let offset = 0
  for (const word of words) {
    offset += (offset > 0 ? 1 : 0) + word.length
    ends.push(offset)
  }
  const total = offset
  const weightSum = weights.reduce((sum, w) => sum + w, 0)

  const cuts: number[] = []
  let share = 0
  for (let k = 0; k < weights.length - 1; k++) {
    share += weights[k]! / weightSum
    const target = share * total
    const earliest = (cuts[cuts.length - 1] ?? -1) + 1
    // Word index after which to cut; the last word can never be a cut.
    const candidates: number[] = []
    for (let i = earliest; i < words.length - 1; i++) candidates.push(i)
    if (candidates.length === 0) break
    const nearest = (pool: number[]): number =>
      pool.reduce((best, i) => (Math.abs(ends[i]! - target) < Math.abs(ends[best]! - target) ? i : best), pool[0]!)
    const within = (pattern: RegExp, reach: number): number[] =>
      candidates.filter((i) => pattern.test(words[i]!) && Math.abs(ends[i]! - target) <= reach * total)
    const sentences = within(SENTENCE_END, SENTENCE_REACH)
    const clauses = within(CLAUSE_END, CLAUSE_REACH)
    cuts.push(nearest(sentences.length > 0 ? sentences : clauses.length > 0 ? clauses : candidates))
  }

  const parts: string[] = []
  let start = 0
  for (const cut of cuts) {
    parts.push(words.slice(start, cut + 1).join(' '))
    start = cut + 1
  }
  parts.push(words.slice(start).join(' '))
  return parts
}

/** Orders a joined balloon's lobes the way its text was read: right to left for vertical Japanese, top to bottom otherwise. */
export function orderLobes(lobes: Box[], verticalColumns: boolean): Box[] {
  const centre = (box: Box) => ({ x: box.x + box.width / 2, y: box.y + box.height / 2 })
  return [...lobes].sort((a, b) =>
    verticalColumns ? centre(b).x - centre(a).x : centre(a).y - centre(b).y,
  )
}

/** A label is a note, not lettering: kept small whatever the region's size. */
export const LABEL_MAX_FONT_SIZE = 16
const LABEL_MIN_WIDTH = 80
const LABEL_MAX_WIDTH = 260

/**
 * Lay out a label for text left on the art: small, about as wide as the
 * region, placed below it, else above, else over its lower edge.
 */
export function fitLabel(
  text: string,
  region: Box,
  page: { width: number; height: number },
  gap: number,
  measure: Measure,
): Fit {
  const width = Math.min(LABEL_MAX_WIDTH, Math.max(LABEL_MIN_WIDTH, region.width * 1.2))
  let fontSize = LABEL_MAX_FONT_SIZE
  let lines = wrapText(text.trim(), width, fontSize, measure)
  while (fontSize > MIN_FONT_SIZE && (lines.length > 3 || widestLine(lines, fontSize, measure) > width)) {
    fontSize--
    lines = wrapText(text.trim(), width, fontSize, measure)
  }
  const blockWidth = Math.min(width, Math.max(0, widestLine(lines, fontSize, measure)))
  const height = blockHeight(lines.length, fontSize)
  const x = Math.min(
    Math.max(0, region.x + (region.width - blockWidth) / 2),
    Math.max(0, page.width - blockWidth),
  )
  const below = region.y + region.height + gap
  const above = region.y - gap - height
  const y =
    below + height + gap <= page.height
      ? below
      : above - gap >= 0
        ? above
        : Math.max(0, region.y + region.height - height)
  return {
    fontSize,
    lines,
    bounds: { x, y, width: blockWidth, height },
    overflowed: widestLine(lines, fontSize, measure) > width,
  }
}
