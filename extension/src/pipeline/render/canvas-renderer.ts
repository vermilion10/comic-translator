import type { Box } from '../detect/types.ts'
import { LINE_HEIGHT, type Measure, fitLabel, fitText, splitText } from './layout.ts'
import { type Field, PLATE_TONE_COVERAGE, type Rgb, fieldAround, plateStyle } from './plate.ts'
import type { RenderInput, RenderResult, RenderedRegion, Renderer, TextBlock } from './types.ts'

/**
 * Render stage on the Canvas 2D API: draws the translation onto the erased
 * image. See layout.ts for fitting and plate.ts for caption plates.
 */

/** No font ships (licensing), so this is a stack of faces likely to be installed. */
const FONT_STACK = '"Comic Sans MS", "Segoe UI", system-ui, sans-serif'

/** A halo behind the glyphs keeps text readable over art: white, or the plate's tone on a plate. */
const HALO_RATIO = 0.22
const HALO_COLOR = '#ffffff'
const INK_COLOR = '#000000'

/**
 * The caption plate: a semi-opaque rounded rect sized to the text block, in
 * the panel's own tone, so the art underneath still shows. plate.ts sets the
 * ink colour and raises the opacity where blending would hurt legibility.
 */
const PLATE_PAD_RATIO = 0.45
const PLATE_RADIUS_RATIO = 0.7

/** Space between a label and the text on art it translates, in pixels. */
const LABEL_GAP = 4

const css = ({ r, g, b }: Rgb, alpha?: number): string => {
  const channel = (value: number): string => Math.round(value).toString()
  const colour = `${channel(r)} ${channel(g)} ${channel(b)}`
  return alpha === undefined ? `rgb(${colour})` : `rgb(${colour} / ${alpha.toString()})`
}

/**
 * Set a translation in a balloon's rectangles. Split across several only when
 * no part gets smaller than the whole text in the first, so a notch in a
 * stepped caption box does not get a word in tiny type.
 */
function layInBalloon(text: string, lobes: Box[], measure: Measure): TextBlock[] {
  const single = fitText(text, lobes[0]!, measure)
  if (lobes.length === 1) return [single]
  const split = splitText(text, lobes.map((lobe) => lobe.width * lobe.height)).map((part, index) =>
    fitText(part, lobes[index]!, measure),
  )
  const smallest = Math.min(...split.map((block) => block.fontSize))
  const overflows = split.some((block) => block.overflowed) && !single.overflowed
  return split.length > 1 && smallest >= single.fontSize && !overflows ? split : [single]
}

export interface CanvasRendererOptions {
  /** Draw the layout rectangle of any region that had to overflow. */
  markOverflow?: boolean
  /** Caption plates behind regions with no flat field to sit on. */
  plates?: boolean
}

export async function createCanvasRenderer(
  options: CanvasRendererOptions = {},
): Promise<Renderer> {
  const platesEnabled = options.plates ?? true

  return Promise.resolve({
    async render(
      erased: ImageBitmap,
      source: ImageBitmap,
      regions: RenderInput[],
    ): Promise<RenderResult> {
      const startedAt = performance.now()

      const canvas = new OffscreenCanvas(erased.width, erased.height)
      const context = canvas.getContext('2d')
      if (!context) {
        throw new Error('could not acquire a 2d context for rendering')
      }

      // The plate test reads the page before erasing, from its own context.
      const probe = new OffscreenCanvas(source.width, source.height)
      const probeContext = probe.getContext('2d', { willReadFrequently: true })
      if (!probeContext) {
        throw new Error('could not acquire a 2d context for the plate probe')
      }
      probeContext.drawImage(source, 0, 0)
      const sourcePixels = probeContext.getImageData(0, 0, source.width, source.height).data

      /** Field under the region on the source page, from the same window the erase stage used. */
      const sampleBackground = (box: Box): Field => fieldAround(sourcePixels, source.width, source.height, box)

      const measure: Measure = (text, fontSize) => {
        context.font = `${fontSize.toString()}px ${FONT_STACK}`
        return context.measureText(text).width
      }

      const laidOut: RenderedRegion[] = regions.map(({ box, text, lobes, label }) => {
        // Text on art left in place: a small solid label in the field's tone.
        if (label) {
          const field = sampleBackground(box)
          const block = fitLabel(text, box, erased, LABEL_GAP, measure)
          return {
            box,
            text,
            fontSize: block.fontSize,
            lines: block.lines,
            bounds: block.bounds,
            overflowed: block.overflowed,
            blocks: [block],
            toneCoverage: field.coverage,
            plateTone: field.ground,
            plated: true,
            labelled: true,
          }
        }
        // Inside a balloon, use its rectangles (wider than the source columns); a
        // balloon is a flat field, so it never needs a plate.
        const inBalloon = lobes !== undefined && lobes.length > 0
        const blocks: TextBlock[] = inBalloon ? layInBalloon(text, lobes, measure) : [fitText(text, box, measure)]
        const first = blocks[0] ?? fitText('', box, measure)
        const field = platesEnabled && !inBalloon ? sampleBackground(box) : null
        return {
          box,
          text,
          fontSize: first.fontSize,
          lines: first.lines,
          bounds: first.bounds,
          overflowed: blocks.some((block) => block.overflowed),
          blocks,
          toneCoverage: field?.coverage ?? 1,
          plateTone: field?.ground ?? null,
          plated: field !== null && field.coverage < PLATE_TONE_COVERAGE,
          labelled: false,
        }
      })

      const laidOutAt = performance.now()

      context.drawImage(erased, 0, 0)
      context.textAlign = 'center'
      context.textBaseline = 'top'
      context.lineJoin = 'round'

      for (const region of laidOut) {
        for (const block of region.blocks) {
          if (block.lines.length === 0) continue

          context.font = `${block.fontSize.toString()}px ${FONT_STACK}`

          // Centre the block vertically inside its bounds.
          const blockHeight = block.lines.length * block.fontSize * LINE_HEIGHT
          const top = block.bounds.y + Math.max(0, (block.bounds.height - blockHeight) / 2)
          const centre = block.bounds.x + block.bounds.width / 2

          let ink = INK_COLOR
          let halo = HALO_COLOR

          if (region.plated && region.plateTone) {
            const tone = region.plateTone
            const style = plateStyle(tone)
            ink = style.ink
            halo = css(tone)
            // A label sits over the original lettering and art, so it is solid.
            const alpha = region.labelled ? 1 : style.alpha

            const widest = block.lines.reduce(
              (widest, line) => Math.max(widest, measure(line, block.fontSize)),
              0,
            )
            const pad = block.fontSize * PLATE_PAD_RATIO
            const plateWidth = widest + pad * 2
            const plateHeight = blockHeight + pad * 2
            const radius = Math.min(
              block.fontSize * PLATE_RADIUS_RATIO,
              plateWidth / 2,
              plateHeight / 2,
            )
            context.save()
            context.fillStyle = css(tone, alpha)
            context.beginPath()
            context.roundRect(centre - plateWidth / 2, top - pad, plateWidth, plateHeight, radius)
            context.fill()
            context.restore()
            context.font = `${block.fontSize.toString()}px ${FONT_STACK}`
          }

          context.strokeStyle = halo
          context.lineWidth = block.fontSize * HALO_RATIO
          context.fillStyle = ink

          block.lines.forEach((line, index) => {
            const y = top + index * block.fontSize * LINE_HEIGHT
            context.strokeText(line, centre, y)
            context.fillText(line, centre, y)
          })

          if (options.markOverflow && block.overflowed) {
            context.save()
            context.strokeStyle = '#ff2d55'
            context.lineWidth = 1.5
            context.strokeRect(
              block.bounds.x,
              block.bounds.y,
              block.bounds.width,
              block.bounds.height,
            )
            context.restore()
          }
        }
      }

      const image = await createImageBitmap(canvas)
      return {
        image,
        regions: laidOut,
        timings: {
          layout: laidOutAt - startedAt,
          draw: performance.now() - laidOutAt,
        },
      }
    },

    async dispose(): Promise<void> {
      return Promise.resolve()
    },
  })
}
