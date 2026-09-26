/** One translated region, paired with the source text it came from. */
export interface Translation {
  /** The OCR text that was translated, unchanged. */
  source: string
  /** The translation, or an empty string when there was nothing to translate. */
  text: string
}

export interface TranslateResult {
  translations: Translation[]
  timings: {
    translate: number
  }
}

/** Called as each region finishes, so a slow multi-region run can show progress. */
export type TranslationListener = (translation: Translation, index: number) => void

/**
 * The translate stage: source strings in, target-language strings out,
 * parallel to the input. The languages are fixed when the translator is built.
 */
export interface Translator {
  translate(
    sources: string[],
    onTranslated?: TranslationListener,
  ): Promise<TranslateResult>
  dispose(): Promise<void>
}
