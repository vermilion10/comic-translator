/**
 * User settings and their defaults. Stored in chrome.storage.local, not sync:
 * sync throttles writes, and the threshold is a slider.
 */

/**
 * The page's script. Set by the user, not guessed: each OCR model outputs its
 * own alphabet whatever it is shown, so a wrong guess gives confident nonsense.
 */
export type SourceLanguage = 'ja' | 'zh' | 'ko'

/**
 * Which translate stage runs. 'local' (web-llm on WebGPU) is free and private;
 * 'cloud' sends the recognised text to Gemini and exists for devices without
 * WebGPU.
 */
export type TranslationProvider = 'local' | 'cloud'

/**
 * The translation target. The local model only does English, so other targets
 * need cloud. Latin-script targets only: the renderer wraps at spaces.
 */
export type TargetLanguage = 'en' | 'id'

export interface Settings {
  /**
   * Minimum detector score for a region to be translated. Detection runs at
   * DETECT_FLOOR and this filters the result, so changing it needs no reload.
   */
  detectThreshold: number
  /** Picks the OCR model. */
  sourceLanguage: SourceLanguage
  /** Which translate stage runs; local unless the user opts in. */
  translationProvider: TranslationProvider
  /**
   * The cloud provider's API key, empty until the user pastes one. Stored
   * unencrypted in chrome.storage.local, which never syncs off the device.
   */
  cloudApiKey: string
  targetLanguage: TargetLanguage
}

export const DEFAULT_SETTINGS: Settings = {
  detectThreshold: 0.6,
  sourceLanguage: 'ja',
  translationProvider: 'local',
  cloudApiKey: '',
  targetLanguage: 'en',
}

export const TRANSLATION_PROVIDERS: { value: TranslationProvider; label: string }[] = [
  { value: 'local', label: 'On this device' },
  { value: 'cloud', label: 'Cloud (Gemini)' },
]

export const TARGET_LANGUAGES: { value: TargetLanguage; label: string }[] = [
  { value: 'en', label: 'English' },
  { value: 'id', label: 'Indonesian' },
]

export const SOURCE_LANGUAGES: { value: SourceLanguage; label: string }[] = [
  { value: 'ja', label: 'Japanese' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ko', label: 'Korean' },
]

/** Detection runs at this floor; the user's threshold filters afterwards. */
export const DETECT_FLOOR = 0.05

export const THRESHOLD_RANGE = { min: 0.05, max: 0.95 } as const

function clampThreshold(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return DEFAULT_SETTINGS.detectThreshold
  }
  return Math.min(THRESHOLD_RANGE.max, Math.max(THRESHOLD_RANGE.min, value))
}

function cleanLanguage(value: unknown): SourceLanguage {
  return SOURCE_LANGUAGES.some((entry) => entry.value === value)
    ? (value as SourceLanguage)
    : DEFAULT_SETTINGS.sourceLanguage
}

function cleanTarget(value: unknown): TargetLanguage {
  return TARGET_LANGUAGES.some((entry) => entry.value === value)
    ? (value as TargetLanguage)
    : DEFAULT_SETTINGS.targetLanguage
}

function cleanProvider(value: unknown): TranslationProvider {
  return TRANSLATION_PROVIDERS.some((entry) => entry.value === value)
    ? (value as TranslationProvider)
    : DEFAULT_SETTINGS.translationProvider
}

/** Trimmed: a stray newline in a header value breaks the request itself. */
function cleanApiKey(value: unknown): string {
  return typeof value === 'string' ? value.trim() : DEFAULT_SETTINGS.cloudApiKey
}

export async function loadSettings(): Promise<Settings> {
  // chrome.storage is typed as an open record, so widen on the way in.
  const stored: Partial<Settings> = await chrome.storage.local.get({
    ...DEFAULT_SETTINGS,
  } as Record<string, unknown>)
  return {
    detectThreshold: clampThreshold(stored.detectThreshold),
    sourceLanguage: cleanLanguage(stored.sourceLanguage),
    translationProvider: cleanProvider(stored.translationProvider),
    cloudApiKey: cleanApiKey(stored.cloudApiKey),
    targetLanguage: cleanTarget(stored.targetLanguage),
  }
}

export async function saveSettings(patch: Partial<Settings>): Promise<void> {
  const next: Record<string, unknown> = {}
  if (patch.detectThreshold !== undefined) {
    next.detectThreshold = clampThreshold(patch.detectThreshold)
  }
  if (patch.sourceLanguage !== undefined) {
    next.sourceLanguage = cleanLanguage(patch.sourceLanguage)
  }
  if (patch.translationProvider !== undefined) {
    next.translationProvider = cleanProvider(patch.translationProvider)
  }
  if (patch.cloudApiKey !== undefined) {
    next.cloudApiKey = cleanApiKey(patch.cloudApiKey)
  }
  if (patch.targetLanguage !== undefined) {
    next.targetLanguage = cleanTarget(patch.targetLanguage)
  }
  await chrome.storage.local.set(next)
}
