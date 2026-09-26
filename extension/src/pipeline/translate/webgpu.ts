/**
 * Whether the local translate stage can run here. `navigator.gpu` alone is not
 * enough: it exists even where WebGPU is disabled or the GPU is blocklisted,
 * and there `requestAdapter()` resolves to null. Checking both avoids a 1.5 GB
 * download that would fail at the first device request.
 */

export type WebGpuSupport =
  | { available: true }
  /** `reason` is for logs, `detail` for the user. */
  | { available: false; reason: 'no-api' | 'no-adapter'; detail: string }

export async function detectWebGpu(): Promise<WebGpuSupport> {
  if (!navigator.gpu) {
    return {
      available: false,
      reason: 'no-api',
      detail: 'this browser does not support WebGPU',
    }
  }

  let adapter: GPUAdapter | null
  try {
    adapter = await navigator.gpu.requestAdapter()
  } catch {
    // A driver failure can throw instead of resolving null; either way there is no adapter.
    adapter = null
  }

  if (!adapter) {
    return {
      available: false,
      reason: 'no-adapter',
      detail: 'this browser supports WebGPU but could not get a GPU adapter',
    }
  }

  return { available: true }
}
