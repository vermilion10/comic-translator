/**
 * Prompt and reply cleanup for the translate stage. A small model does not
 * reliably answer with the bare translation, so the cleanup is load-bearing.
 */

import type { SourceLanguage, TargetLanguage } from '../settings.ts'

const SOURCE_NAME: Record<SourceLanguage, { name: string; romanised: string }> = {
  ja: { name: 'Japanese', romanised: 'romaji' },
  zh: { name: 'Chinese', romanised: 'pinyin' },
  ko: { name: 'Korean', romanised: 'romanization' },
}

const TARGET_NAME: Record<TargetLanguage, string> = {
  en: 'English',
  id: 'Indonesian',
}

/** Instructions for one line. Japanese into English equals SYSTEM_PROMPT. */
export function systemPrompt(source: SourceLanguage, target: TargetLanguage): string {
  const from = SOURCE_NAME[source]
  const into = TARGET_NAME[target]
  return [
    `You are a manga translator. Translate the ${from.name} line the user gives you into natural ${into}.`,
    `Reply with the ${into} translation only: no notes, no ${from.romanised}, no quotation marks, no explanation.`,
    'Keep it short and in the register a character would actually speak.',
    `If the line is only sound effects or punctuation, reply with the closest ${into} equivalent.`,
  ].join(' ')
}

/** Japanese into English: the local stage's prompt. */
export const SYSTEM_PROMPT = systemPrompt('ja', 'en')

/** Labels a small model prefixes despite instructions; anchored so a colon in the text survives. */
const LABEL = /^\s*(?:english|translation|translated(?:\s+text)?|answer|output)\s*[:：-]\s*/i

/** Matched as a pair so a line that merely ends in a quote is left alone. */
const WRAPPING_QUOTES: [string, string][] = [
  ['"', '"'],
  ["'", "'"],
  ['「', '」'],
  ['『', '』'],
  ['“', '”'],
  ['‘', '’'],
]

function stripWrappingQuotes(text: string): string {
  for (const [open, close] of WRAPPING_QUOTES) {
    if (text.length > open.length + close.length &&
        text.startsWith(open) && text.endsWith(close)) {
      return text.slice(open.length, -close.length).trim()
    }
  }
  return text
}

/** Reduce a raw reply to the translation: first non-empty line, labels and wrapping quotes removed. */
export function cleanTranslation(raw: string): string {
  const firstLine = raw
    .split('\n')
    .map((line) => line.trim())
    .find((line) => line.length > 0)
  if (firstLine === undefined) return ''

  let text = firstLine.replace(LABEL, '').trim()
  text = stripWrappingQuotes(text)
  return text.trim()
}

/** True when there is nothing worth spending a model call on. */
export function isTranslatable(source: string): boolean {
  return source.trim().length > 0
}

/**
 * Batching, for the cloud stage only: the free tier's rate limit rejects one
 * request per bubble, so a page goes as one numbered request.
 */
export function batchSystemPrompt(source: SourceLanguage, target: TargetLanguage): string {
  return [
    systemPrompt(source, target),
    'The user will give you several numbered lines from one page.',
    'Reply with one numbered line per input line, in the same order,',
    'formatted exactly as "1. translation" with nothing else.',
    'Keep every number, even if a line is only a sound effect.',
  ].join(' ')
}

export const BATCH_SYSTEM_PROMPT = batchSystemPrompt('ja', 'en')

export function formatBatch(sources: string[]): string {
  return sources.map((source, index) => `${(index + 1).toString()}. ${source}`).join('\n')
}

/**
 * Split a numbered reply into one translation per input line. Entries the
 * model skipped are undefined (not empty) so the caller can re-ask for them.
 * Unnumbered lines continue the entry above.
 */
export function parseBatch(raw: string, count: number): (string | undefined)[] {
  const out = new Array<string | undefined>(count)
  let current: number | null = null

  for (const line of raw.split('\n')) {
    const numbered = /^\s*(\d{1,3})\s*[.):：]\s*(.*)$/.exec(line)
    if (numbered) {
      const index = Number(numbered[1]) - 1
      if (index >= 0 && index < count) {
        current = index
        out[index] = numbered[2] ?? ''
        continue
      }
      // Out of range: treat as continuation text rather than dropping it.
    }
    if (current !== null && line.trim().length > 0) {
      out[current] = `${out[current] ?? ''} ${line.trim()}`.trim()
    }
  }

  return out.map((entry) => (entry === undefined ? undefined : cleanTranslation(entry)))
}
