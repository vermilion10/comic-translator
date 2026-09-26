import type { SourceLanguage, TargetLanguage } from '../settings.ts'
import {
  batchSystemPrompt,
  systemPrompt,
  cleanTranslation,
  formatBatch,
  isTranslatable,
  parseBatch,
} from './prompt.ts'
import type { TranslateResult, Translation, TranslationListener, Translator } from './types.ts'

/**
 * Translate stage backed by Gemini's REST API: the cloud option for devices
 * that cannot run the local one. Gemini has a real free API tier, which keeps
 * the extension usable without a card. Prompts are shared with the local stage
 * (prompt.ts).
 */

/**
 * Flash-Lite: cheapest on the free tier and does not think by default. Pinned
 * to a version, not the -latest alias, so a page does not translate
 * differently between sessions.
 */
export const DEFAULT_GEMINI_MODEL = 'gemini-3.5-flash-lite'

const ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models'

/**
 * A runaway guard. Generous because on a model that thinks, reasoning tokens
 * count against it and an exhausted budget returns a truncated or empty reply.
 */
const MAX_TOKENS = 512

/** Waits after a 429, in order. The limit is per minute, so the last wait crosses a minute boundary. */
const RETRY_DELAYS = [2000, 8000, 30000]

/** Regions per request: a guard against a pathological page, not a routine path. */
const BATCH_LIMIT = 40

export interface GeminiTranslatorOptions {
  apiKey: string
  model?: string
  /** Defaults to Japanese into English, the measured pair. */
  source?: SourceLanguage
  target?: TargetLanguage
}

interface GeminiResponse {
  candidates?: {
    content?: { parts?: { text?: string }[] }
    finishReason?: string
  }[]
  promptFeedback?: { blockReason?: string }
  error?: { message?: string; status?: string }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/** Turn an HTTP failure into something the user can act on: a bad key and a rate limit need different fixes. */
async function describeFailure(response: Response): Promise<string> {
  let detail = ''
  try {
    const body = (await response.json()) as GeminiResponse
    detail = body.error?.message ?? ''
  } catch {
    // A non-JSON body from a proxy or gateway; the status still says enough.
  }

  if (response.status === 400 && /api.?key/i.test(detail)) {
    return 'the Gemini API key was rejected. Check it in the extension options.'
  }
  if (response.status === 401 || response.status === 403) {
    return 'the Gemini API key was refused for this request. Check it in the extension options.'
  }
  if (response.status === 429) {
    return 'the Gemini free tier rate limit was reached. Wait a minute and try again.'
  }
  if (response.status >= 500) {
    return `Gemini returned ${response.status.toString()}. That is their side; try again.`
  }
  return `Gemini returned ${response.status.toString()}${detail ? `: ${detail}` : ''}`
}

export function createGeminiTranslator(options: GeminiTranslatorOptions): Translator {
  const model = options.model ?? DEFAULT_GEMINI_MODEL
  const apiKey = options.apiKey.trim()
  const single = systemPrompt(options.source ?? 'ja', options.target ?? 'en')
  const batch = batchSystemPrompt(options.source ?? 'ja', options.target ?? 'en')

  if (!apiKey) {
    // Thrown at construction, so a missing key is reported before any OCR runs.
    throw new Error(
      'cloud translation is selected but no API key is set. Add one in the extension options.',
    )
  }

  const url = `${ENDPOINT}/${encodeURIComponent(model)}:generateContent`

  async function requestOnce(system: string, user: string, tokens: number): Promise<Response> {
    return fetch(url, {
      method: 'POST',
      headers: {
        // The header, not ?key=, which ends up in logs.
        'x-goog-api-key': apiKey,
        'content-type': 'application/json',
      },
      credentials: 'omit',
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: system }] },
        contents: [{ role: 'user', parts: [{ text: user }] }],
        generationConfig: {
          // Hosted inference is not reproducible even at temperature 0; do not rely on a stable answer.
          temperature: 0,
          maxOutputTokens: tokens,
          // No thinkingConfig, deliberately. Asking for thinkingBudget 0 looks
          // like the obvious way to keep a one-line translation cheap, and it
          // is not portable: gemini-3.5-flash-lite and gemini-flash-lite-latest
          // both reject it outright with HTTP 400 'Request contains an invalid
          // argument', while gemini-3.1-flash-lite and the non-lite flash
          // models accept it. Omitting it works on all of them, and the lite
          // models do not think unless asked, so there is nothing to switch
          // off on the model that ships. MAX_TOKENS is what protects the reply
          // on a model that does think.
        },
      }),
    })
  }

  /** One request, with the rate-limit retry. Returns the raw reply text. */
  async function ask(system: string, user: string, tokens: number): Promise<string> {
    let response = await requestOnce(system, user, tokens)
    for (let attempt = 0; attempt < RETRY_DELAYS.length && response.status === 429; attempt += 1) {
      await sleep(RETRY_DELAYS[attempt] ?? 0)
      response = await requestOnce(system, user, tokens)
    }
    if (!response.ok) {
      throw new Error(await describeFailure(response))
    }

    const body = (await response.json()) as GeminiResponse
    const candidate = body.candidates?.[0]

    // A safety block is a 200 with no text; report it rather than typeset an empty bubble.
    if (body.promptFeedback?.blockReason) {
      throw new Error(
        `Gemini declined to translate a line (${body.promptFeedback.blockReason}).`,
      )
    }

    const text = candidate?.content?.parts?.map((part) => part.text ?? '').join('') ?? ''
    if (!text && candidate?.finishReason && candidate.finishReason !== 'STOP') {
      throw new Error(`Gemini returned no text (${candidate.finishReason}).`)
    }
    return text
  }

  return {
    async translate(
      sources: string[],
      onTranslated?: TranslationListener,
    ): Promise<TranslateResult> {
      const startedAt = performance.now()
      const texts = new Array<string>(sources.length).fill('')

      // Empty OCR results are not sent: numbering them invites the model to invent text.
      const wanted = sources
        .map((source, index) => ({ source, index }))
        .filter((entry) => isTranslatable(entry.source))

      for (let start = 0; start < wanted.length; start += BATCH_LIMIT) {
        const chunk = wanted.slice(start, start + BATCH_LIMIT)

        if (chunk.length === 1) {
          // One line needs no numbering, which would risk a stray "1." on the page.
          const entry = chunk[0]
          if (entry) {
            texts[entry.index] = cleanTranslation(
              await ask(single, entry.source, MAX_TOKENS),
            )
          }
          continue
        }

        const reply = await ask(
          batch,
          formatBatch(chunk.map((entry) => entry.source)),
          MAX_TOKENS * chunk.length,
        )
        const parsed = parseBatch(reply, chunk.length)

        for (const [position, entry] of chunk.entries()) {
          const text = parsed[position]
          if (text !== undefined) {
            texts[entry.index] = text
            continue
          }
          // The model skipped this number: ask again for just this line.
          texts[entry.index] = cleanTranslation(
            await ask(single, entry.source, MAX_TOKENS),
          )
        }
      }

      // Reported after the batch, in order, since a batch has no per-line moment.
      const translations: Translation[] = sources.map((source, index) => ({
        source,
        text: texts[index] ?? '',
      }))
      for (const [index, translation] of translations.entries()) {
        onTranslated?.(translation, index)
      }

      return {
        translations,
        timings: { translate: performance.now() - startedAt },
      }
    },

    async dispose(): Promise<void> {
      // Nothing to release: no weights, no GPU, no socket held open. Present
      // because Translator requires it and the runner disposes uniformly.
    },
  }
}
