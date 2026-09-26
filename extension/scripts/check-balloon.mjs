/**
 * Checks src/pipeline/inpaint/balloon.ts against ml/eval/balloon.py: same
 * fallback decision, pixels, colour and layout rectangles on every region.
 *
 * Usage:
 *   python ml/scripts/dump_balloon_fixture.py
 *   node scripts/check-balloon.mjs [fixture.json]
 */

import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { clipToBalloon } from '../src/pipeline/inpaint/balloon.ts'

const here = dirname(fileURLToPath(import.meta.url))
const fixturePath = process.argv[2] ?? join(here, '..', '..', 'ml', 'eval', 'data', 'balloon_fixture.json')
const cases = JSON.parse(readFileSync(fixturePath, 'utf8'))

const bytes = (text) => new Uint8Array(Buffer.from(text, 'base64'))
const failures = []
let filled = 0
let lobeCount = 0

for (const [n, c] of cases.entries()) {
  const rgb = bytes(c.rgb)
  const rgba = new Uint8ClampedArray(c.width * c.height * 4)
  for (let i = 0; i < c.width * c.height; i++) {
    rgba[i * 4] = rgb[i * 3]
    rgba[i * 4 + 1] = rgb[i * 3 + 1]
    rgba[i * 4 + 2] = rgb[i * 3 + 2]
    rgba[i * 4 + 3] = 255
  }
  // The window is the whole page here; the clip recomputes it from the region's bounds.
  const patch = (data) => ({ x: 0, y: 0, width: c.width, height: c.height, data })
  const got = clipToBalloon(rgba, c.width, c.height, patch(bytes(c.base)), patch(bytes(c.seed)))
  const name = `#${n} ${c.post} ${c.detector}`

  if (got === null || c.expected === null) {
    if ((got === null) !== (c.expected === null)) {
      failures.push(`${name}: ${got ? 'fills' : 'falls back'}, Python ${c.expected ? 'fills' : 'falls back'}`)
    }
    continue
  }
  filled++
  let pixels = 0
  let indexSum = 0
  for (let y = 0; y < got.patch.height; y++) {
    for (let x = 0; x < got.patch.width; x++) {
      if (!got.patch.data[y * got.patch.width + x]) continue
      pixels++
      indexSum += (got.patch.y + y) * c.width + got.patch.x + x
    }
  }
  const color = [got.color.r, got.color.g, got.color.b]
  if (pixels !== c.expected.pixels || indexSum !== c.expected.indexSum) {
    failures.push(`${name}: ${pixels} px, Python ${c.expected.pixels}`)
  } else if (color.join() !== c.expected.color.join()) {
    failures.push(`${name}: colour ${color.join()}, Python ${c.expected.color.join()}`)
  } else {
    const lobes = got.lobes.map((l) => [l.x, l.y, l.width, l.height])
    const same =
      lobes.length === c.expected.lobes.length &&
      lobes.every((l, i) => l.every((v, j) => Math.abs(v - c.expected.lobes[i][j]) < 1e-9))
    if (!same) failures.push(`${name}: lobes ${JSON.stringify(lobes)}, Python ${JSON.stringify(c.expected.lobes)}`)
    lobeCount += lobes.length
  }
}

console.log(`${cases.length} regions compared, ${filled} of them filled, ${lobeCount} lobes`)
if (failures.length) {
  console.log(failures.slice(0, 20).join('\n'))
  console.log(`${failures.length} mismatches`)
  process.exit(1)
}
console.log('balloon clip matches the Python side')
