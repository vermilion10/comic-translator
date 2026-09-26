import type { Box } from '../detect/types.ts'
import { type BalloonFill, clipToBalloon } from './balloon.ts'
import { addPatch, regionPatch, seedPatch } from './mask.ts'
import { fieldAround, isTextOnArt } from '../render/plate.ts'
import { DEFAULT_RADIUS } from './opencv-telea.ts'
import type {
  EraseRegion,
  InpaintOptions,
  InpaintResult,
  Inpainter,
  SandboxRequest,
  SandboxResponse,
} from './types.ts'

/**
 * Inpaint stage backed by OpenCV.js's Telea algorithm, which runs in a
 * sandboxed page (src/ui/sandbox). This half builds the mask, sends the
 * buffers to the frame and turns the answer back into an ImageBitmap.
 */

const SANDBOX_PAGE = 'src/ui/sandbox/inpaint.html'

/** The sandbox loads a 12.7 MB module before it answers. */
const READY_TIMEOUT_MS = 60_000

export interface TeleaInpainterOptions {
  /** How far Telea looks outside the mask for a fill colour. */
  radius?: number
  /** Pixels to grow each box by before erasing; 0 works best (mask.ts). */
  padding?: number
}

export async function createTeleaInpainter(
  options: TeleaInpainterOptions = {},
): Promise<Inpainter> {
  const radius = options.radius ?? DEFAULT_RADIUS

  const frame = document.createElement('iframe')
  frame.src = chrome.runtime.getURL(SANDBOX_PAGE)
  frame.hidden = true
  frame.setAttribute('aria-hidden', 'true')

  const pending = new Map<number, (response: SandboxResponse) => void>()
  let nextId = 0

  const onMessage = (event: MessageEvent<SandboxResponse>) => {
    // The sandbox's origin is opaque, so check the frame identity instead.
    if (event.source !== frame.contentWindow) return

    const response = event.data
    if (response.type === 'inpaint-ready') {
      pending.get(-1)?.(response)
      return
    }
    // id -1 is the startup channel, not a job.
    pending.get(response.id)?.(response)
  }
  window.addEventListener('message', onMessage)

  const ready = new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`inpaint sandbox did not load within ${READY_TIMEOUT_MS}ms`))
    }, READY_TIMEOUT_MS)
    // A startup failure arrives on the same id, so it surfaces instead of timing out.
    pending.set(-1, (response) => {
      clearTimeout(timer)
      pending.delete(-1)
      if (response.type === 'inpaint-error') {
        reject(new Error(`inpaint sandbox failed to start: ${response.message}`))
      } else {
        resolve()
      }
    })
  })

  document.body.append(frame)
  try {
    await ready
  } catch (error) {
    window.removeEventListener('message', onMessage)
    frame.remove()
    throw error
  }

  function send(request: SandboxRequest): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      pending.set(request.id, (response) => {
        pending.delete(request.id)
        if (response.type === 'inpaint-result') resolve(response.rgba)
        else if (response.type === 'inpaint-error') reject(new Error(response.message))
      })
      frame.contentWindow?.postMessage(request, '*', [request.rgba, request.mask])
    })
  }

  return {
    async inpaint(
      source: ImageBitmap,
      regions: EraseRegion[],
      { balloonFill = false, keepTextOnArt = false }: InpaintOptions = {},
    ): Promise<InpaintResult> {
      const startedAt = performance.now()

      const canvas = new OffscreenCanvas(source.width, source.height)
      const context = canvas.getContext('2d')
      if (!context) {
        throw new Error('could not acquire a 2d context for inpainting')
      }
      context.drawImage(source, 0, 0)
      const imageData = context.getImageData(0, 0, source.width, source.height)
      const { width, height } = source

      // Each region is painted flat when balloonFill finds its balloon, otherwise
      // sent to Telea. Decided on the source pixels.
      const mask = new Uint8Array(width * height)
      const fills: BalloonFill[] = []
      const lobes: Box[][] = regions.map(() => [])
      const onArt: boolean[] = regions.map(() => false)
      let toTelea = 0
      for (const [index, region] of regions.entries()) {
        const patch = regionPatch(width, height, region, options.padding)
        if (!patch) continue
        const seed = balloonFill ? seedPatch(width, height, region) : null
        const fill = seed ? clipToBalloon(imageData.data, width, height, patch, seed) : null
        if (fill) {
          fills.push(fill)
          lobes[index] = fill.lobes
        } else if (keepTextOnArt && isTextOnArt(fieldAround(imageData.data, width, height, region.box))) {
          onArt[index] = true
        } else {
          addPatch(mask, width, patch)
          toTelea++
        }
      }

      const maskedAt = performance.now()

      // Nothing for Telea: skip the round trip and paint on a copy.
      const painted =
        toTelea === 0
          ? imageData.data.buffer.slice(0)
          : await send({
              type: 'inpaint-request',
              id: nextId++,
              width,
              height,
              rgba: imageData.data.buffer,
              mask: mask.buffer,
              radius,
            })

      const pixels = new Uint8ClampedArray(painted)
      for (const { patch, color } of fills) {
        for (let y = 0; y < patch.height; y++) {
          for (let x = 0; x < patch.width; x++) {
            if (!patch.data[y * patch.width + x]) continue
            const i = ((patch.y + y) * width + patch.x + x) * 4
            pixels[i] = color.r
            pixels[i + 1] = color.g
            pixels[i + 2] = color.b
          }
        }
      }

      const image = await createImageBitmap(new ImageData(pixels, width, height))
      return {
        image,
        timings: { mask: maskedAt - startedAt, inpaint: performance.now() - maskedAt },
        filled: fills.length,
        lobes,
        onArt,
      }
    },

    async dispose(): Promise<void> {
      window.removeEventListener('message', onMessage)
      pending.clear()
      frame.remove()
      return Promise.resolve()
    },
  }
}
