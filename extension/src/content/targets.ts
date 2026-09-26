import type { Box } from '../pipeline/detect/types.ts'
import { findStrip } from './tiles.ts'

/**
 * Finds the image to translate without the page's cooperation. Nothing here
 * relies on an event the page can cancel (some readers cancel contextmenu).
 * A point is resolved through the whole hit stack, into shadow roots, since
 * the top element is often an overlay. Candidates are ranked by how much of the
 * viewport they cover, so the usual case needs no pointer input.
 */

/**
 * Displayed size an image must reach before the control offers it. Set for
 * recall: a missed comic is worse than an offer on a large photo.
 */
export const MIN_SHORT_SIDE = 300
export const MIN_LONG_SIDE = 400

function clearsGate(width: number, height: number): boolean {
  return Math.min(width, height) >= MIN_SHORT_SIDE && Math.max(width, height) >= MIN_LONG_SIDE
}

export interface ImageTarget {
  image: HTMLImageElement
  /** Area of the image inside the viewport right now, in CSS pixels. */
  visibleArea: number
}

/**
 * Big enough to be a comic page, and actually loaded: lazy viewers stretch a
 * 1x1 placeholder across the tile's full box.
 */
export function hasPixels(image: HTMLImageElement): boolean {
  return image.complete && image.naturalWidth > 1 && image.naturalHeight > 1
}

export function isCandidate(image: HTMLImageElement): boolean {
  if (!hasPixels(image)) return false
  const rect = image.getBoundingClientRect()
  return clearsGate(rect.width, rect.height)
}

/**
 * A tile of a split page, judged by its strip: the tile's width by the whole
 * strip's height. Only the height requirement is relaxed, so nothing narrow
 * (a column of avatars, a banner) passes by being joined up.
 */
export function isStripCandidate(image: HTMLImageElement, images: HTMLImageElement[]): boolean {
  if (!hasPixels(image)) return false
  const rect = image.getBoundingClientRect()
  if (clearsGate(rect.width, rect.height)) return true
  const strip = findStrip(image, images)
  return strip !== null && clearsGate(rect.width, strip.height)
}

/** How much of the image is on screen, which is what "the page you are reading" means. */
export function visibleArea(image: HTMLImageElement): number {
  const rect = image.getBoundingClientRect()
  const width = Math.max(0, Math.min(rect.right, window.innerWidth) - Math.max(rect.left, 0))
  const height = Math.max(0, Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0))
  return width * height
}

/** Every img in the document, including inside open shadow roots. */
export function collectImages(root: ParentNode = document): HTMLImageElement[] {
  const found: HTMLImageElement[] = []
  for (const element of root.querySelectorAll('*')) {
    if (element instanceof HTMLImageElement) found.push(element)
    if (element.shadowRoot) found.push(...collectImages(element.shadowRoot))
  }
  return found
}

function largest(
  images: Iterable<HTMLImageElement>,
  offerable: (image: HTMLImageElement) => boolean,
): ImageTarget | null {
  let best: ImageTarget | null = null
  for (const image of images) {
    if (!offerable(image)) continue
    const onScreen = visibleArea(image)
    if (onScreen <= 0) continue
    if (!best || onScreen > best.visibleArea) best = { image, visibleArea: onScreen }
  }
  return best
}

/**
 * The largest on-screen candidate, or null. Whole images win over slices; the
 * strip pass, which is expensive, runs only when there is no whole image.
 */
export function bestTarget(
  onScreen: Iterable<HTMLImageElement>,
  images: HTMLImageElement[] = [],
): ImageTarget | null {
  const whole = largest(onScreen, isCandidate)
  if (whole || images.length === 0) return whole
  return largest(onScreen, (image) => isStripCandidate(image, images))
}

/** The image under a point, walking the hit stack past overlays and into shadow roots. */
export function imageAtPoint(x: number, y: number, ignore?: Element): HTMLImageElement | null {
  const entered = new Set<ShadowRoot>()
  let root: DocumentOrShadowRoot = document

  for (;;) {
    const stack = root.elementsFromPoint(x, y)
    let descend: ShadowRoot | null = null

    for (const element of stack) {
      if (ignore?.contains(element) ?? false) continue
      if (element instanceof HTMLImageElement) return element
      const { shadowRoot } = element
      if (!descend && shadowRoot && !entered.has(shadowRoot)) descend = shadowRoot
    }

    if (!descend) return null
    entered.add(descend)
    root = descend
  }
}

/** A rectangle in page coordinates, which is where a drag is recorded. */
export interface PageRect {
  left: number
  top: number
  width: number
  height: number
}

export interface SourceMapping {
  /** The rectangle in source-image pixels, clamped to the image. */
  box: Box
  /** True when the drawn rectangle reached past the image and was cut back. */
  clipped: boolean
}

/**
 * Where an image's pixels are painted, in page coordinates: inside the border
 * and padding, letterboxed by object-fit. Transforms are not handled.
 */
function paintedRect(image: HTMLImageElement): PageRect | null {
  const rect = image.getBoundingClientRect()
  const style = window.getComputedStyle(image)
  const edge = (property: string): number => Number.parseFloat(style.getPropertyValue(property)) || 0

  const left = rect.left + window.scrollX + edge('border-left-width') + edge('padding-left')
  const top = rect.top + window.scrollY + edge('border-top-width') + edge('padding-top')
  const width =
    rect.width -
    edge('border-left-width') -
    edge('border-right-width') -
    edge('padding-left') -
    edge('padding-right')
  const height =
    rect.height -
    edge('border-top-width') -
    edge('border-bottom-width') -
    edge('padding-top') -
    edge('padding-bottom')
  if (width <= 0 || height <= 0) return null

  const natural = { width: image.naturalWidth, height: image.naturalHeight }
  if (natural.width <= 0 || natural.height <= 0) return null

  const cover = Math.max(width / natural.width, height / natural.height)
  const contain = Math.min(width / natural.width, height / natural.height)
  let scaleX = width / natural.width
  let scaleY = height / natural.height
  switch (style.objectFit) {
    case 'contain':
      scaleX = scaleY = contain
      break
    case 'cover':
      scaleX = scaleY = cover
      break
    case 'none':
      scaleX = scaleY = 1
      break
    case 'scale-down':
      scaleX = scaleY = Math.min(1, contain)
      break
    default:
      // 'fill', the initial value: the two axes stretch independently.
      break
  }

  const painted = { width: natural.width * scaleX, height: natural.height * scaleY }
  // object-position (default centred); percentages resolve against the leftover space.
  const [alongX = '50%', alongY = alongX] = style.objectPosition.split(/\s+/)
  const share = (value: string, free: number): number => {
    if (value.endsWith('%')) return (free * Number.parseFloat(value)) / 100
    if (value === 'left' || value === 'top') return 0
    if (value === 'right' || value === 'bottom') return free
    if (value === 'center') return free / 2
    const offset = Number.parseFloat(value)
    return Number.isNaN(offset) ? free / 2 : offset
  }

  return {
    left: left + share(alongX, width - painted.width),
    top: top + share(alongY, height - painted.height),
    width: painted.width,
    height: painted.height,
  }
}

/**
 * Map a rectangle drawn over the page into source-image pixels, the space
 * every pipeline stage uses. An overhanging rectangle is clamped, not rejected.
 */
export function toSourceBox(image: HTMLImageElement, drawn: PageRect): SourceMapping | null {
  const painted = paintedRect(image)
  if (!painted) return null

  const scaleX = image.naturalWidth / painted.width
  const scaleY = image.naturalHeight / painted.height

  const rawLeft = (drawn.left - painted.left) * scaleX
  const rawTop = (drawn.top - painted.top) * scaleY
  const rawRight = rawLeft + drawn.width * scaleX
  const rawBottom = rawTop + drawn.height * scaleY

  const left = Math.max(0, Math.min(image.naturalWidth, rawLeft))
  const top = Math.max(0, Math.min(image.naturalHeight, rawTop))
  const right = Math.max(0, Math.min(image.naturalWidth, rawRight))
  const bottom = Math.max(0, Math.min(image.naturalHeight, rawBottom))
  if (right - left <= 0 || bottom - top <= 0) return null

  return {
    box: {
      x: Math.round(left),
      y: Math.round(top),
      width: Math.round(right - left),
      height: Math.round(bottom - top),
    },
    clipped:
      rawLeft < -1 || rawTop < -1 ||
      rawRight > image.naturalWidth + 1 ||
      rawBottom > image.naturalHeight + 1,
  }
}

/** How much of `drawn` lands on `image`, in page-coordinate square pixels. */
export function overlapArea(image: HTMLImageElement, drawn: PageRect): number {
  const rect = image.getBoundingClientRect()
  const left = rect.left + window.scrollX
  const top = rect.top + window.scrollY
  const width = Math.max(0, Math.min(left + rect.width, drawn.left + drawn.width) - Math.max(left, drawn.left))
  const height = Math.max(0, Math.min(top + rect.height, drawn.top + drawn.height) - Math.max(top, drawn.top))
  return width * height
}
