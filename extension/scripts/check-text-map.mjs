/**
 * Checks src/pipeline/detect/text-map.ts against ml/textseg/detector.py: the
 * same boxes from the same probability maps, and the same erase-mask pixels.
 *
 * Usage:
 *   python ml/scripts/dump_text_map_fixture.py --model <textseg.onnx>
 *   node scripts/check-text-map.mjs [fixture.json]
 */

import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { decodeTextMap } from '../src/pipeline/detect/text-map.ts'
import { buildMask } from '../src/pipeline/inpaint/mask.ts'

const here = dirname(fileURLToPath(import.meta.url))
const fixturePath = process.argv[2] ?? join(here, '..', '..', 'ml', 'eval', 'data', 'text_map_fixture.json')

// numpy sums probabilities in float32 in its own order, so scores get a tolerance.
const BOX_TOLERANCE = 1e-6
const SCORE_TOLERANCE = 1e-5

const fixture = JSON.parse(readFileSync(fixturePath, 'utf8'))
let compared = 0
let masks = 0
const failures = []

for (const page of fixture.pages) {
  const bytes = Buffer.from(page.prob, 'base64')
  const prob = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)

  const shaped = decodeTextMap(prob, page.mapSize, page.layout, page.width, page.height, fixture.threshold)
  const mask = buildMask(page.width, page.height, shaped)
  let pixels = 0
  let indexSum = 0
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) {
      pixels++
      indexSum += i
    }
  }
  masks++
  if (pixels !== page.eraseMask.pixels || indexSum !== page.eraseMask.indexSum) {
    failures.push(`${page.post} erase mask: ${pixels} px, Python has ${page.eraseMask.pixels}`)
  }
  for (const expected of page.expected) {
    const found = decodeTextMap(prob, page.mapSize, page.layout, page.width, page.height, fixture.threshold, {
      binarize: expected.binarize,
      unshrinkScale: expected.unshrinkScale,
    })
    const setting = `binarize ${expected.binarize} x${expected.unshrinkScale}`
    // Order by position: near-equal scores can sort differently on the two sides.
    const byPosition = (a, b) => a.box[0] - b.box[0] || a.box[1] - b.box[1]
    const got = found
      .map((d) => ({ box: [d.box.x, d.box.y, d.box.width, d.box.height], score: d.score }))
      .sort(byPosition)
    const want = [...expected.detections].sort(byPosition)
    compared += want.length
    if (got.length !== want.length) {
      failures.push(`${page.post} ${setting}: ${got.length} boxes, Python has ${want.length}`)
      continue
    }
    for (const [i, g] of got.entries()) {
      const w = want[i]
      const boxOff = Math.max(...g.box.map((v, j) => Math.abs(v - w.box[j])))
      const scoreOff = Math.abs(g.score - w.score)
      if (boxOff > BOX_TOLERANCE || scoreOff > SCORE_TOLERANCE) {
        failures.push(
          `${page.post} ${setting} box ${i}: off by ${boxOff.toExponential(2)} px, score ${scoreOff.toExponential(2)}`,
        )
      }
    }
  }
}

console.log(`${fixture.pages.length} pages, ${compared} boxes and ${masks} erase masks compared`)
if (failures.length) {
  console.log(failures.slice(0, 20).join('\n'))
  console.log(`${failures.length} mismatches`)
  process.exit(1)
}
console.log('decoder and erase mask match the Python side')
