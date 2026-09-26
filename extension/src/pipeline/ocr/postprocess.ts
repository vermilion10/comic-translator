/**
 * Port of `post_process` in manga_ocr/ocr.py. Without it the model's half-width
 * ASCII never matches manga-ocr's full-width reference outputs.
 */

/** Offset from a half-width ASCII code point to its full-width twin. */
const FULLWIDTH_OFFSET = 0xfee0

export function postProcess(text: string): string {
  // Neither rule can fire with the current vocabulary; kept so a swapped-in
  // vocabulary does not silently change the output.
  let result = text.replace(/\s+/gu, '').replaceAll('…', '...')

  // A run of katakana middle dots is the model's way of writing an ellipsis.
  result = result.replace(/[・.]{2,}/gu, (run) => '.'.repeat(run.length))

  return result.replace(/[!-~]/gu, (char) =>
    String.fromCodePoint(char.codePointAt(0)! + FULLWIDTH_OFFSET),
  )
}
