import type { SourceLanguage } from '../settings.ts'
import { createMangaOcr } from './manga-ocr.ts'
import { createPaddleOcr } from './paddle-ocr.ts'
import type { Recognizer } from './types.ts'

/**
 * Picks the OCR model for the source language. The engines are not
 * interchangeable: manga-ocr garbles Chinese and cannot write hangul, and
 * PP-OCR cannot read vertical Japanese, so this is a switch, not a fallback.
 */
export async function createRecognizerFor(language: SourceLanguage): Promise<Recognizer> {
  if (language === 'zh' || language === 'ko') {
    const korean = language === 'ko'
    return createPaddleOcr({
      detUrl: chrome.runtime.getURL('models/ppocr-det.onnx'),
      recUrl: chrome.runtime.getURL(korean ? 'models/ppocr-rec-ko.onnx' : 'models/ppocr-rec.onnx'),
      dictUrl: chrome.runtime.getURL(
        korean ? 'models/ppocr-dict-ko.txt' : 'models/ppocr-dict.txt',
      ),
      // Korean spaces its words; Chinese does not.
      lineSeparator: korean ? ' ' : '',
    })
  }
  return createMangaOcr({
    encoderUrl: chrome.runtime.getURL('models/manga-ocr-encoder.onnx'),
    decoderUrl: chrome.runtime.getURL('models/manga-ocr-decoder.onnx'),
    vocabUrl: chrome.runtime.getURL('models/manga-ocr-vocab.txt'),
    lineDetectorUrl: chrome.runtime.getURL('models/ppocr-det.onnx'),
  })
}

/** Shown while loading, since the models differ by about 90 MB. */
export function recognizerLabel(language: SourceLanguage): string {
  if (language === 'zh') return 'Chinese OCR model (31 MB)'
  if (language === 'ko') return 'Korean OCR model (23 MB)'
  return 'Japanese OCR model (126 MB)'
}
