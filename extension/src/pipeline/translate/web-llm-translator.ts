import type { MLCEngine } from '@mlc-ai/web-llm'

import { SYSTEM_PROMPT, cleanTranslation, isTranslatable } from './prompt.ts'
import type { TranslateResult, Translation, TranslationListener, Translator } from './types.ts'
import { detectWebGpu } from './webgpu.ts'

/**
 * Translate stage backed by web-llm on WebGPU. Weights download at runtime
 * from Hugging Face (permissive CORS, so no host permission) and are cached
 * in the Cache API.
 */

/**
 * Gemma 2 2B, instruction-tuned for Japanese, 4-bit with f16 activations; it
 * translated noticeably better than Qwen2.5-1.5B at 1.5 GB against 0.9 GB.
 * Needs the WebGPU "shader-f16" feature.
 */
export const DEFAULT_MODEL_ID = 'gemma-2-2b-jpn-it-q4f16_1-MLC'

/** Bubble lines are short; this is a runaway guard, not a target length. */
const MAX_TOKENS = 96

export interface WebLlmTranslatorOptions {
  modelId?: string
  /** Reports weight download and GPU upload progress, 0..1. */
  onProgress?: (progress: number, text: string) => void
}

export async function createWebLlmTranslator(
  options: WebLlmTranslatorOptions = {},
): Promise<Translator> {
  const modelId = options.modelId ?? DEFAULT_MODEL_ID

  // The harness builds this directly, so check again here.
  const support = await detectWebGpu()
  if (!support.available) {
    throw new Error(
      `on-device translation needs WebGPU and ${support.detail}. ` +
        'Cloud translation covers this case; turn it on in the extension options.',
    )
  }

  // Imported lazily: the library is a 6 MB chunk.
  const webllm = await import('@mlc-ai/web-llm')

  let engine: MLCEngine
  try {
    engine = await webllm.CreateMLCEngine(modelId, {
      initProgressCallback: (report) => {
        options.onProgress?.(report.progress, report.text)
      },
    })
  } catch (error) {
    // Most commonly the GPU refusing to allocate the weights.
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`could not load ${modelId}: ${message}`, { cause: error })
  }

  return {
    async translate(
      sources: string[],
      onTranslated?: TranslationListener,
    ): Promise<TranslateResult> {
      const startedAt = performance.now()
      const translations: Translation[] = []

      for (const [index, source] of sources.entries()) {
        let text = ''
        if (isTranslatable(source)) {
          const completion = await engine.chat.completions.create({
            messages: [
              { role: 'system', content: SYSTEM_PROMPT },
              { role: 'user', content: source },
            ],
            // Deterministic, so a page translates the same way every time.
            temperature: 0,
            max_tokens: MAX_TOKENS,
            stream: false,
          })
          text = cleanTranslation(completion.choices[0]?.message.content ?? '')
        }

        const translation: Translation = { source, text }
        translations.push(translation)
        onTranslated?.(translation, index)
      }

      return {
        translations,
        timings: { translate: performance.now() - startedAt },
      }
    },

    async dispose(): Promise<void> {
      await engine.unload()
    },
  }
}
