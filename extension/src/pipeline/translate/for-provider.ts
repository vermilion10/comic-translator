import type { Settings } from '../settings.ts'
import { DEFAULT_GEMINI_MODEL, createGeminiTranslator } from './gemini-translator.ts'
import type { Translator } from './types.ts'
import { createWebLlmTranslator } from './web-llm-translator.ts'
import { detectWebGpu } from './webgpu.ts'

/**
 * Picks the translate stage and reports why one cannot run, as a value the
 * caller can offer a fix for instead of failing after the page was erased.
 * Local is the default: free, and the text never leaves the device.
 */

/** A setup problem the user can fix on the options page; see setup-needed.ts. */
export type SetupReason = 'no-webgpu' | 'no-api-key' | 'cloud-only-target'

export type TranslatorChoice =
  | { ok: true; translator: Translator; label: string }
  /** `reason` is what the UI branches on; every reason is fixed on the options page. */
  | { ok: false; reason: SetupReason; message: string }

export interface ProviderOptions {
  /** Forwarded to web-llm, which reports weight download and GPU upload. */
  onProgress?: (progress: number, text: string) => void
}

export async function createTranslatorFor(
  settings: Settings,
  options: ProviderOptions = {},
): Promise<TranslatorChoice> {
  if (settings.translationProvider === 'cloud') {
    if (!settings.cloudApiKey) {
      return {
        ok: false,
        reason: 'no-api-key',
        message:
          'Cloud translation is selected but no API key is saved. ' +
          'Add a Gemini API key in the extension options.',
      }
    }
    return {
      ok: true,
      translator: createGeminiTranslator({
        apiKey: settings.cloudApiKey,
        source: settings.sourceLanguage,
        target: settings.targetLanguage,
      }),
      label: `cloud translation (${DEFAULT_GEMINI_MODEL})`,
    }
  }

  // The local model only does English. Never switch to cloud on the user's
  // behalf: that would send their text to a third party they did not choose.
  if (settings.targetLanguage !== 'en') {
    return {
      ok: false,
      reason: 'cloud-only-target',
      message:
        'Translating into anything but English needs cloud translation; the ' +
        'on-device model only does English. Switch to cloud in the options, or set ' +
        'the target language back to English.',
    }
  }

  // Checked before downloading 1.5 GB of weights, not after.
  const support = await detectWebGpu()
  if (!support.available) {
    return {
      ok: false,
      reason: 'no-webgpu',
      message:
        `On-device translation needs WebGPU, and ${support.detail}. ` +
        'Everything else on this page worked. Cloud translation can finish it, ' +
        'but it sends the text off this device.',
    }
  }

  return {
    ok: true,
    translator: await createWebLlmTranslator({ onProgress: options.onProgress }),
    label: 'on-device translation',
  }
}

/** Whether a translator built for `before` must be rebuilt for `after`. */
export function translatorIsStale(before: Settings, after: Settings): boolean {
  return (
    before.translationProvider !== after.translationProvider ||
    before.cloudApiKey !== after.cloudApiKey ||
    before.sourceLanguage !== after.sourceLanguage ||
    before.targetLanguage !== after.targetLanguage
  )
}
