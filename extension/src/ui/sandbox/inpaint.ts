import { inpaintRgba, type OpenCvLike } from '../../pipeline/inpaint/opencv-telea.ts'
import type { SandboxRequest, SandboxResponse } from '../../pipeline/inpaint/types.ts'

/**
 * The sandboxed host for OpenCV.js, whose embind runtime needs `new Function`,
 * which only a sandboxed page may use. It has no chrome.* APIs: pixels in,
 * pixels out.
 *
 * OpenCV.js loads as a classic script tag, not an import: it is CommonJS whose
 * export is a Promise, and the bundler's interop breaks awaiting it.
 */

const TAG = '[manga-translator:sandbox]'

/** Set by /vendor/opencv.js; resolves once the wasm runtime has initialised. */
declare const cv: Promise<OpenCvLike> | undefined

function post(message: SandboxResponse, transfer: Transferable[] = []): void {
  // This frame's origin is opaque, so no specific targetOrigin is possible.
  window.parent.postMessage(message, '*', transfer)
}

async function main(): Promise<void> {
  if (typeof cv === 'undefined') {
    throw new Error('/vendor/opencv.js did not load; run npm run assets')
  }
  // A real Promise this time, so awaiting it is safe.
  const opencv = await cv

  window.addEventListener('message', (event: MessageEvent<SandboxRequest>) => {
    const request = event.data
    if (request?.type !== 'inpaint-request') return

    try {
      const rgba = new Uint8ClampedArray(request.rgba)
      const painted = inpaintRgba(
        opencv,
        rgba,
        request.width,
        request.height,
        new Uint8Array(request.mask),
        request.radius,
      )
      // Not a SharedArrayBuffer: this page is not cross-origin isolated.
      const buffer = painted.buffer as ArrayBuffer
      post({ type: 'inpaint-result', id: request.id, rgba: buffer }, [buffer])
    } catch (error) {
      post({
        type: 'inpaint-error',
        id: request.id,
        message: error instanceof Error ? error.message : String(error),
      })
    }
  })

  console.log(`${TAG} OpenCV.js ready`)
  post({ type: 'inpaint-ready' })
}

// Report a startup failure instead of letting the host time out.
main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error)
  console.error(`${TAG} failed to start`, error)
  post({ type: 'inpaint-error', id: -1, message })
})
