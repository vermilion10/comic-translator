import { createTextMapDetector } from '../../pipeline/detect/text-map-detector.ts'
import type { Box, Detector } from '../../pipeline/detect/types.ts'
import { createTeleaInpainter } from '../../pipeline/inpaint/telea-inpainter.ts'
import type { EraseRegion, Inpainter } from '../../pipeline/inpaint/types.ts'
import { createRecognizerFor, recognizerLabel } from '../../pipeline/ocr/for-language.ts'
import type { Recognizer } from '../../pipeline/ocr/types.ts'
import { createCanvasRenderer } from '../../pipeline/render/canvas-renderer.ts'
import { orderLobes } from '../../pipeline/render/layout.ts'
import type { Renderer } from '../../pipeline/render/types.ts'
import {
  DETECT_FLOOR,
  type Settings,
  type SourceLanguage,
} from '../../pipeline/settings.ts'
import {
  type SetupReason,
  createTranslatorFor,
  translatorIsStale,
} from '../../pipeline/translate/for-provider.ts'
import type { Translator } from '../../pipeline/translate/types.ts'
import { PIPELINE_PORT, type RunnerRequest, type RunnerResponse } from './protocol.ts'

/**
 * The whole pipeline, in one offscreen document shared by every tab. Not in
 * the content script (the page's storage origin and CSP) and not in the
 * service worker (no DOM). Being top-level, it loads each model once per
 * session and can be cross-origin isolated for onnxruntime threads. It also
 * fetches the image, since an extension-origin document is exempt from CORS.
 *
 * A request may carry a drawn region instead of asking for detection.
 */

const TAG = '[manga-translator:runner]'

let detector: Detector | null = null
let recognizer: Recognizer | null = null
/** Which script `recognizer` reads, so a changed setting rebuilds it. */
let recognizerLanguage: SourceLanguage | null = null
let inpainter: Inpainter | null = null
let translator: Translator | null = null
/** Which settings `translator` was built for, so a changed one rebuilds it. */
let translatorSettings: Settings | null = null
let renderer: Renderer | null = null

/**
 * The last few rendered pages, by original image URL, so a drawn-box run
 * continues from the translated page instead of starting over. Capped, since
 * each holds a bitmap copy.
 */
const RETAINED_RENDERS = 3
const retained = new Map<string, ImageBitmap>()

function retain(srcUrl: string, image: ImageBitmap): void {
  retained.get(srcUrl)?.close()
  retained.delete(srcUrl)
  retained.set(srcUrl, image)
  while (retained.size > RETAINED_RENDERS) {
    const [oldest] = retained.keys()
    if (oldest === undefined) break
    retained.get(oldest)?.close()
    retained.delete(oldest)
  }
}

/** Trim a box to the image, so a rectangle drawn past an edge is still usable. */
function clampBox(box: Box, width: number, height: number): Box | null {
  const x = Math.max(0, Math.min(width, Math.round(box.x)))
  const y = Math.max(0, Math.min(height, Math.round(box.y)))
  const right = Math.max(0, Math.min(width, Math.round(box.x + box.width)))
  const bottom = Math.max(0, Math.min(height, Math.round(box.y + box.height)))
  if (right - x < 1 || bottom - y < 1) return null
  return { x, y, width: right - x, height: bottom - y }
}

/** The ports of translations in flight, by request id. */
const ports = new Map<number, chrome.runtime.Port>()

function post(message: RunnerResponse): void {
  const port = ports.get(message.id)
  // The tab may have closed mid-run.
  try {
    port?.postMessage(message)
  } catch {
    ports.delete(message.id)
  }
}

/** A failure the user can fix in settings, so the control can offer the fix. */
class SetupNeeded extends Error {
  constructor(
    readonly reason: SetupReason,
    message: string,
  ) {
    super(message)
    this.name = 'SetupNeeded'
  }
}

function progress(id: number, detail: string): void {
  console.log(`${TAG} ${detail}`)
  post({ type: 'runner-progress', id, detail })
}

/**
 * Fetch and decode the image. Credentials are omitted: with broad host
 * permissions, a credentialed fetch would let any page read a user's private
 * images from another site.
 */
async function loadImage(id: number, srcUrl: string): Promise<ImageBitmap> {
  progress(id, 'Fetching image…')
  const response = await fetch(srcUrl, { credentials: 'omit' })
  if (!response.ok) {
    throw new Error(`could not fetch the image: ${response.status.toString()}`)
  }
  return createImageBitmap(await response.blob())
}

async function translateImage(
  id: number,
  srcUrl: string,
  settings: Settings,
  manual?: Box,
): Promise<void> {

  // A manual run continues from the translated page; a full run starts from the original.
  const carried = manual ? retained.get(srcUrl) : undefined
  const source = carried ? await createImageBitmap(carried) : await loadImage(id, srcUrl)

  // Detections are erased by their text shape; a drawn box is erased whole.
  let regions: EraseRegion[]
  if (manual) {
    const box = clampBox(manual, source.width, source.height)
    if (!box) {
      source.close()
      throw new Error('the region you drew does not overlap the image')
    }
    progress(id, 'Using the region you drew…')
    console.log(
      `${TAG} manual region ${box.width.toString()}x${box.height.toString()} at ` +
        `${box.x.toString()},${box.y.toString()}` +
        `${carried ? ', continuing from the previous result' : ''}`,
    )
    regions = [{ box }]
  } else {
    progress(id, 'Loading detector…')
    // Built at the fixed floor; the user's threshold filters the result.
    detector ??= await createTextMapDetector({
      modelUrl: chrome.runtime.getURL('models/textseg.onnx'),
      scoreThreshold: DETECT_FLOOR,
    })

    progress(id, 'Detecting text…')
    const detected = await detector.detect(source)
    regions = detected.detections.filter((detection) => detection.score >= settings.detectThreshold)
    console.log(
      `${TAG} ${detected.detections.length.toString()} detections, ` +
        `${regions.length.toString()} above threshold ${settings.detectThreshold.toFixed(2)}`,
    )
    if (regions.length === 0) {
      throw new Error(
        `no text regions above the confidence threshold (${settings.detectThreshold.toFixed(2)}), ` +
          `try drawing a box around the text instead`,
      )
    }
  }

  // A new source language needs a different model; release the old one.
  if (recognizer && recognizerLanguage !== settings.sourceLanguage) {
    await recognizer.dispose()
    recognizer = null
  }
  if (!recognizer) {
    progress(id, `Loading ${recognizerLabel(settings.sourceLanguage)}…`)
    recognizer = await createRecognizerFor(settings.sourceLanguage)
    recognizerLanguage = settings.sourceLanguage
  }

  const boxes: Box[] = regions.map((region) => region.box)
  progress(id, `Reading ${boxes.length.toString()} regions…`)
  const read = await recognizer.recognize(source, boxes, (_, index) => {
    progress(id, `Reading ${(index + 1).toString()}/${boxes.length.toString()}…`)
  })

  progress(id, 'Erasing…')
  inpainter ??= await createTeleaInpainter()
  // Regions in a flat balloon are painted with its colour; text on art is left for labels.
  const erased = await inpainter.inpaint(source, regions, { balloonFill: true, keepTextOnArt: true })

  // The cloud translator has the key baked in, so settings changes rebuild it.
  if (translator && translatorSettings && translatorIsStale(translatorSettings, settings)) {
    await translator.dispose()
    translator = null
    translatorSettings = null
  }
  if (!translator) {
    progress(id, 'Loading translation…')
    const choice = await createTranslatorFor(settings, {
      // Pass web-llm's own text too: it reports 0% for the whole GPU upload.
      onProgress: (fraction, text) => {
        progress(id, `Translation model ${(fraction * 100).toFixed(0)}% ${text}`)
      },
    })
    if (!choice.ok) {
      // Release the erased image, then report a setup problem the control can offer a fix for.
      erased.image.close()
      source.close()
      throw new SetupNeeded(choice.reason, choice.message)
    }
    translator = choice.translator
    translatorSettings = settings
    console.log(`${TAG} translating with ${choice.label}`)
  }

  progress(id, 'Translating…')
  const translated = await translator.translate(
    read.recognitions.map((recognition) => recognition.text),
    (_, index) => {
      progress(id, `Translating ${(index + 1).toString()}/${boxes.length.toString()}…`)
    },
  )

  progress(id, 'Typesetting…')
  renderer ??= await createCanvasRenderer()
  const rendered = await renderer.render(
    erased.image,
    source,
    boxes.map((box, index) => ({
      box,
      text: translated.translations[index]?.text ?? '',
      // Set inside the balloon, in reading order.
      lobes: orderLobes(erased.lobes[index] ?? [], settings.sourceLanguage === 'ja'),
      label: erased.onArt[index] ?? false,
    })),
  )

  erased.image.close()
  source.close()
  // Copied before the port send, which consumes the bitmap.
  retain(srcUrl, await createImageBitmap(rendered.image))
  const image = await toDataUrl(rendered.image)
  rendered.image.close()
  post({ type: 'runner-result', id, image, regions: boxes.length })
}

/** A port serialises to JSON, so the result travels as a PNG data URL. */
async function toDataUrl(bitmap: ImageBitmap): Promise<string> {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height)
  canvas.getContext('2d')?.drawImage(bitmap, 0, 0)
  const blob = await canvas.convertToBlob({ type: 'image/png' })
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      resolve(reader.result as string)
    }
    reader.onerror = () => {
      reject(reader.error ?? new Error('could not encode the translated image'))
    }
    reader.readAsDataURL(blob)
  })
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== PIPELINE_PORT) return

  port.onMessage.addListener((request: RunnerRequest) => {
    if (request.type !== 'runner-translate') return
    ports.set(request.id, port)
    port.onDisconnect.addListener(() => ports.delete(request.id))

    void translateImage(request.id, request.srcUrl, request.settings, request.box)
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error)
        console.error(`${TAG} pipeline failed`, error)
        post({
          type: 'runner-error',
          id: request.id,
          message,
          ...(error instanceof SetupNeeded ? { reason: error.reason } : {}),
        })
      })
      .finally(() => ports.delete(request.id))
  })
})

console.log(`${TAG} ready, ${String(self.crossOriginIsolated)} cross-origin isolation`)
void chrome.runtime.sendMessage({ type: 'pipeline-ready', payload: {} })
