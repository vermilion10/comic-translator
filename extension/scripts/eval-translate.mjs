/**
 * Runs the cloud translate stage over reference OCR lines (kha-white's 12
 * Japanese crops, ml/eval/{zh,ko}_reference.json) so the output can be read.
 * Imports gemini-translator.ts directly, so it must stay free of chrome APIs.
 *
 * Usage:
 *   node scripts/eval-translate.mjs --key-file <path> [--model <id>]... [--per-line] [--target en|id]
 *   node scripts/eval-translate.mjs --key-file <path> --json out.json
 */

import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { createGeminiTranslator } from '../src/pipeline/translate/gemini-translator.ts'

const here = dirname(fileURLToPath(import.meta.url))
const evalDir = join(here, '..', '..', 'ml', 'eval')

const JA_REFERENCE =
  'https://raw.githubusercontent.com/kha-white/manga-ocr/master/tests/data/expected_results.json'

/** Which reference lines to run: a handful per language, few enough to read. */
const KO_INDICES = [0, 1, 2, 3, 4, 5]
const ZH_INDICES = [0, 1, 2, 3, 4]

async function japaneseLines() {
  const response = await fetch(JA_REFERENCE)
  if (!response.ok) throw new Error(`could not fetch the Japanese set: ${response.status}`)
  const cases = await response.json()
  return cases.map((entry) => ({
    language: 'ja',
    id: entry.filename.replace(/\.jpg$/, ''),
    source: entry.result,
  }))
}

function paddleLines(language, indices) {
  const cases = JSON.parse(readFileSync(join(evalDir, `${language}_reference.json`), 'utf-8'))
  return indices
    .filter((index) => index < cases.length)
    .map((index) => ({
      language,
      id: `${language}${String(index).padStart(2, '0')}`,
      source: cases[index].text,
    }))
}

function parseArgs(argv) {
  const models = []
  let keyFile = null
  let json = null
  let perLine = false
  let target = 'en'
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--model') models.push(argv[(i += 1)])
    else if (argv[i] === '--key-file') keyFile = argv[(i += 1)]
    else if (argv[i] === '--json') json = argv[(i += 1)]
    else if (argv[i] === '--per-line') perLine = true
    else if (argv[i] === '--target') target = argv[(i += 1)]
    else throw new Error(`unknown argument ${argv[i]}`)
  }
  if (!keyFile) throw new Error('--key-file is required')
  if (models.length === 0) models.push('gemini-2.5-flash-lite')
  return { models, keyFile, json, perLine, target }
}

const { models, keyFile, json, perLine, target } = parseArgs(process.argv.slice(2))
const apiKey = readFileSync(keyFile, 'utf-8').trim()

const lines = [
  ...(await japaneseLines()),
  ...paddleLines('zh', ZH_INDICES),
  ...paddleLines('ko', KO_INDICES),
]

console.log(`${lines.length} lines, ${models.length} model(s)\n`)

const results = {}

for (const model of models) {
  const outputs = []
  let failed = 0
  const startedAt = Date.now()

  // One translator per source language, as the extension builds one per page.
  for (const language of ['ja', 'zh', 'ko']) {
    const group = lines.filter((line) => line.language === language)
    const translator = createGeminiTranslator({ apiKey, model, source: language, target })
    if (perLine) {
      for (const line of group) {
        try {
          const result = await translator.translate([line.source])
          outputs.push({ ...line, text: result.translations[0].text })
        } catch (error) {
          failed += 1
          outputs.push({ ...line, text: '', error: error.message })
        }
      }
    } else {
      try {
        const result = await translator.translate(group.map((line) => line.source))
        group.forEach((line, i) => { outputs.push({ ...line, text: result.translations[i].text }) })
      } catch (error) {
        failed += group.length
        for (const line of group) outputs.push({ ...line, text: '', error: error.message })
      }
    }
  }

  const elapsed = Date.now() - startedAt
  results[model] = { outputs, failed, elapsedMs: elapsed }

  console.log(`===== ${model}  (${(elapsed / lines.length).toFixed(0)} ms/line, ${failed} failed)`)
  for (const out of outputs) {
    console.log(`  ${out.id} ${out.language}  ${out.source}`)
    console.log(`        -> ${out.error ? `FAILED: ${out.error}` : out.text}`)
  }
  console.log()
}

if (json) {
  writeFileSync(json, JSON.stringify({ lines, results }, null, 1), 'utf-8')
  console.log(`wrote ${json}`)
}
