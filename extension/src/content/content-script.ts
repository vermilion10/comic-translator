import type { Box } from '../pipeline/detect/types.ts'
import { type Message, type ResponseFor, sendMessage } from '../pipeline/messages'
import { loadSettings } from '../pipeline/settings.ts'
import { PIPELINE_PORT, type RunnerRequest, type RunnerResponse } from '../ui/runner/protocol'
import { mountControl } from './control.ts'
import { SetupNeededError } from './setup-needed.ts'

/**
 * Page integration: find the image, send its URL to the shared pipeline
 * document, show progress, and swap the result in. No pipeline stage runs
 * here, since a content script shares the page's storage origin and CSP.
 *
 * The context menu and the floating control (control.ts) both call translate();
 * a drawn region is the same call with a box.
 */

const TAG = '[manga-translator:content]'

/** Set on an <img> whose src we replaced, holding the original URL. */
const ORIGINAL_SRC = 'mangaTranslatorOriginal'

console.log(`${TAG} content script loaded on ${location.href}`)

let nextRequestId = 0

/** Run one request through the shared pipeline document, which the background worker creates on first use. */
async function runPipeline(
  request: RunnerRequest,
  onProgress: (detail: string) => void,
): Promise<RunnerResponse> {
  const { ready } = await sendMessage('ensure-pipeline', {})
  if (!ready) throw new Error('the translation pipeline did not start')

  return new Promise<RunnerResponse>((resolve, reject) => {
    const port = chrome.runtime.connect({ name: PIPELINE_PORT })
    let settled = false
    port.onMessage.addListener((response: RunnerResponse) => {
      if (response.id !== request.id) return
      if (response.type === 'runner-progress') {
        onProgress(response.detail)
        return
      }
      settled = true
      resolve(response)
      port.disconnect()
    })
    port.onDisconnect.addListener(() => {
      if (!settled) reject(new Error('the translation pipeline stopped before answering'))
    })
    port.postMessage(request)
  })
}

/** Minimal status pinned over the image, since a run takes a while. */
function showBadge(image: HTMLImageElement, text: string): HTMLElement {
  let badge = image.nextElementSibling as HTMLElement | null
  if (!badge?.classList.contains('manga-translator-badge')) {
    badge = document.createElement('div')
    badge.className = 'manga-translator-badge'
    badge.style.cssText =
      'position:absolute;z-index:2147483647;background:rgba(0,0,0,.82);color:#fff;' +
      'font:12px/1.4 system-ui,sans-serif;padding:6px 10px;border-radius:6px;' +
      'pointer-events:none;max-width:60vw;'
    image.after(badge)
  }
  const rect = image.getBoundingClientRect()
  badge.style.left = `${(rect.left + window.scrollX + 8).toString()}px`
  badge.style.top = `${(rect.top + window.scrollY + 8).toString()}px`
  badge.textContent = text
  return badge
}

function clearBadge(image: HTMLImageElement): void {
  const badge = image.nextElementSibling
  if (badge?.classList.contains('manga-translator-badge')) badge.remove()
}

/** The context menu reports the image URL, not the element; srcUrl matches currentSrc for srcset images. */
function findImage(srcUrl: string): HTMLImageElement | null {
  const images = [...document.images]
  return (
    images.find((image) => image.currentSrc === srcUrl || image.src === srcUrl) ??
    images.find((image) => image.dataset[ORIGINAL_SRC] === srcUrl) ??
    null
  )
}

/**
 * Run the pipeline over one image and swap the result in. Status goes to the
 * badge and to the caller; failures are shown and rethrown. `box` replaces
 * detection with a region the user drew.
 */
async function translate(
  image: HTMLImageElement,
  onStatus: (text: string) => void = () => undefined,
  box?: Box,
): Promise<void> {
  const report = (text: string): void => {
    showBadge(image, text)
    onStatus(text)
  }

  report('Starting…')
  try {
    const id = nextRequestId++
    // Send the URL, not pixels: the page's CORS rules block a content script's
    // fetch. Always the original URL, which the pipeline keys its result on.
    const srcUrl = image.dataset[ORIGINAL_SRC] ?? (image.currentSrc || image.src)
    const settings = await loadSettings()
    const request: RunnerRequest = { type: 'runner-translate', id, srcUrl, box, settings }

    const result = await runPipeline(request, report)

    if (result.type === 'runner-error') {
      // A reason means the user can fix it; the control offers the fix.
      throw result.reason
        ? new SetupNeededError(result.reason, result.message)
        : new Error(result.message)
    }
    if (result.type !== 'runner-result') throw new Error('the pipeline answered with nothing')

    // A blob URL keeps the DOM small, unlike a data URL in src.
    const blob = await (await fetch(result.image)).blob()

    // Keep the original so the restore action has something to go back to.
    image.dataset[ORIGINAL_SRC] ??= image.currentSrc || image.src
    image.srcset = ''
    image.src = URL.createObjectURL(blob)
    clearBadge(image)
    console.log(`${TAG} replaced image, ${result.regions.toString()} regions`)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    console.error(`${TAG} translation failed`, error)
    showBadge(image, `Failed: ${message}`)
    setTimeout(() => { clearBadge(image) }, 6000)
    throw error instanceof Error ? error : new Error(message)
  }
}

function isTranslated(image: HTMLImageElement): boolean {
  return image.dataset[ORIGINAL_SRC] !== undefined
}

function restoreImage(image: HTMLImageElement): boolean {
  const original = image.dataset[ORIGINAL_SRC]
  if (original === undefined) return false

  if (image.src.startsWith('blob:')) URL.revokeObjectURL(image.src)
  image.src = original
  delete image.dataset[ORIGINAL_SRC]
  clearBadge(image)
  return true
}

chrome.runtime.onMessage.addListener((message: Message, _sender, sendResponse) => {
  switch (message.type) {
    case 'translate-image': {
      const image = findImage(message.payload.srcUrl)
      if (image) {
        void translate(image).catch(() => undefined)
      } else {
        console.warn(`${TAG} could not find the clicked image on the page`)
      }
      const response: ResponseFor<'translate-image'> = { accepted: image !== null }
      sendResponse(response)
      break
    }
    case 'restore-image': {
      const image = findImage(message.payload.srcUrl)
      const response: ResponseFor<'restore-image'> = {
        restored: image !== null && restoreImage(image),
      }
      sendResponse(response)
      break
    }
    default:
      break
  }
  return false
})

mountControl({
  translate: (image, onStatus, box) => translate(image, onStatus, box),
  restore: restoreImage,
  isTranslated,
})

async function announce(): Promise<void> {
  try {
    const { receivedAt } = await sendMessage('content-script-ready', {
      url: location.href,
    })
    console.log(`${TAG} background acknowledged at ${new Date(receivedAt).toISOString()}`)
  } catch (error) {
    console.error(`${TAG} background did not respond`, error)
  }
}

void announce()
