import * as ort from 'onnxruntime-web/wasm'

/**
 * Threads for onnxruntime-web in this document. They need SharedArrayBuffer,
 * which only cross-origin isolated documents get (the manifest's COOP/COEP
 * covers the offscreen pipeline and the harness); elsewhere this uses one.
 * Four threads read an OCR region about 2.5x faster than one; more would
 * compete with the page the user is reading.
 */
const MAX_THREADS = 4

export function configureOrtThreads(): number {
  const threads = globalThis.crossOriginIsolated
    ? Math.max(1, Math.min(MAX_THREADS, navigator.hardwareConcurrency || 1))
    : 1
  ort.env.wasm.numThreads = threads
  return threads
}
