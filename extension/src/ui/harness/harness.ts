import type { Box, Detection, DetectResult, Detector } from '../../pipeline/detect/types.ts'
import { createTextMapDetector } from '../../pipeline/detect/text-map-detector.ts'
import { createTeleaInpainter } from '../../pipeline/inpaint/telea-inpainter.ts'
import type { Inpainter } from '../../pipeline/inpaint/types.ts'
import { createRecognizerFor, recognizerLabel } from '../../pipeline/ocr/for-language.ts'
import type { Recognition, Recognizer } from '../../pipeline/ocr/types.ts'
import { createCanvasRenderer } from '../../pipeline/render/canvas-renderer.ts'
import type { Rgb } from '../../pipeline/render/plate.ts'
import { orderLobes } from '../../pipeline/render/layout.ts'
import type { Renderer } from '../../pipeline/render/types.ts'
import {
  type Settings,
  type SourceLanguage,
  TARGET_LANGUAGES,
  type TargetLanguage,
  TRANSLATION_PROVIDERS,
  type TranslationProvider,
  loadSettings,
  saveSettings,
} from '../../pipeline/settings.ts'
import { createTranslatorFor, translatorIsStale } from '../../pipeline/translate/for-provider.ts'
import type { Translation, Translator } from '../../pipeline/translate/types.ts'

const TAG = '[manga-translator:harness]'

/** Detect at the floor and filter here, so moving the slider needs no new inference. */
const FLOOR = 0.05

/** Mean token log-probability below which a reading is likely hallucinated on a textless crop. */
const WEAK_READING = -0.5

const dropZone = requireElement<HTMLDivElement>('drop')
const filePicker = requireElement<HTMLInputElement>('file')
const controls = requireElement<HTMLDivElement>('controls')
const threshold = requireElement<HTMLInputElement>('threshold')
const thresholdValue = requireElement<HTMLOutputElement>('threshold-value')
const ocrToggle = requireElement<HTMLInputElement>('ocr')
const languageSelect = requireElement<HTMLSelectElement>('language')
const status = requireElement<HTMLDivElement>('status')
const canvas = requireElement<HTMLCanvasElement>('canvas')
const cleanToggle = requireElement<HTMLInputElement>('clean')
const shapeToggle = requireElement<HTMLInputElement>('shape')
const balloonToggle = requireElement<HTMLInputElement>('balloon')
const labelToggle = requireElement<HTMLInputElement>('label-art')
const cleanedView = requireElement<HTMLElement>('cleaned-view')
const cleanedCanvas = requireElement<HTMLCanvasElement>('cleaned')
const translateToggle = requireElement<HTMLInputElement>('translate')
const providerSelect = requireElement<HTMLSelectElement>('provider')
const targetSelect = requireElement<HTMLSelectElement>('target')
const openOptions = requireElement<HTMLButtonElement>('open-options')
const renderedView = requireElement<HTMLElement>('rendered-view')
const renderedCanvas = requireElement<HTMLCanvasElement>('rendered')
const platedView = requireElement<HTMLElement>('plated-view')
const platedCanvas = requireElement<HTMLCanvasElement>('plated')
const readingsTable = requireElement<HTMLTableElement>('readings')
const readingsBody = requireElement<HTMLTableSectionElement>('readings-body')
const benchLines = requireElement<HTMLTextAreaElement>('lines')
const benchRun = requireElement<HTMLButtonElement>('bench-run')
const benchStatus = requireElement<HTMLElement>('bench-status')
const benchTable = requireElement<HTMLTableElement>('bench-table')
const benchBody = requireElement<HTMLTableSectionElement>('bench-body')

let detector: Detector | null = null
let recognizer: Recognizer | null = null
let recognizerLanguage: SourceLanguage | null = null
let inpainter: Inpainter | null = null
let translator: Translator | null = null
/** The settings `translator` was built for, to know when it is stale. */
let translatorSettings: Settings | null = null
/** Two renderers, plates off and on, shown side by side. */
let renderer: Renderer | null = null
let plateRenderer: Renderer | null = null
let lastImage: ImageBitmap | null = null
let lastResult: DetectResult | null = null

/** Keyed by detection index, so a re-filter reuses readings. */
let readings = new Map<number, Recognition>()
let reading = false
let translations = new Map<number, Translation>()
let translating = false
let cleaning = false
/** Boxes the cleaned canvas currently reflects, to avoid redundant passes. */
let cleanedFor: string | null = null
/** Kept rather than closed, because the render stage draws on top of it. */
let cleanedImage: ImageBitmap | null = null
/** Balloon layout rectangles from the last erase, keyed like `readings`. */
let cleanedLobes = new Map<number, Box[]>()
/** Regions the last erase left in place as text on art, keyed the same way. */
let cleanedOnArt = new Set<number>()
let renderedFor: string | null = null
let rendering = false
/** Plate decision per detection index, for the table. Keyed like `readings`. */
let plates = new Map<number, { plated: boolean; toneCoverage: number; tone: Rgb | null }>()

/** Bumped per dropped image, so a slow OCR pass cannot write into a newer one. */
let generation = 0

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id)
  if (!element) throw new Error(`harness markup is missing #${id}`)
  return element as T
}

/** The tone a plate was filled with, short enough for a table cell. */
function describeTone({ r, g, b }: Rgb): string {
  return [r, g, b].map((channel) => Math.round(channel).toString()).join(',')
}

/** Built once and reused; loading and warming a session is the slow part. */
async function getDetector(): Promise<Detector> {
  if (detector) return detector

  status.textContent = 'Loading detector…'
  detector = await createTextMapDetector({
    modelUrl: chrome.runtime.getURL('models/textseg.onnx'),
    scoreThreshold: FLOOR,
  })
  return detector
}

function currentLanguage(): SourceLanguage {
  const value = languageSelect.value
  return value === 'zh' || value === 'ko' ? value : 'ja'
}

async function getRecognizer(): Promise<Recognizer> {
  const language = currentLanguage()
  if (recognizer && recognizerLanguage === language) return recognizer
  if (recognizer) await recognizer.dispose()

  status.textContent = `Loading ${recognizerLabel(language)}…`
  recognizer = await createRecognizerFor(language)
  recognizerLanguage = language
  return recognizer
}

async function getInpainter(): Promise<Inpainter> {
  if (inpainter) return inpainter

  status.textContent = 'Loading OpenCV.js (12.7 MB, first run only)…'
  inpainter = await createTeleaInpainter()
  return inpainter
}

/**
 * The translator the settings select, chosen as the runner chooses it. Returns
 * null, with the reason on screen, when it cannot run.
 */
async function getTranslator(): Promise<Translator | null> {
  const settings = await loadSettings()
  if (translator && translatorSettings && translatorIsStale(translatorSettings, settings)) {
    await translator.dispose()
    translator = null
    translatorSettings = null
  }
  if (translator) return translator

  status.textContent = 'Loading translator…'
  const choice = await createTranslatorFor(settings, {
    onProgress: (progress, text) => {
      status.textContent = `Translation model ${(progress * 100).toFixed(0)}%: ${text}`
    },
  })
  if (!choice.ok) {
    status.textContent = choice.message
    return null
  }
  console.log(`${TAG} translating with ${choice.label}`)
  translator = choice.translator
  translatorSettings = settings
  return translator
}

/** The translate stage alone over pasted lines, for comparing providers on the same input. */
let benchTranslator: Translator | null = null
let benchSettings: Awaited<ReturnType<typeof loadSettings>> | null = null

async function runBench(): Promise<void> {
  const sources = benchLines.value
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)

  if (sources.length === 0) {
    benchStatus.textContent = 'Nothing to translate.'
    return
  }

  benchRun.disabled = true
  try {
    const settings = await loadSettings()
    if (benchTranslator && benchSettings && translatorIsStale(benchSettings, settings)) {
      await benchTranslator.dispose()
      benchTranslator = null
    }
    if (!benchTranslator) {
      benchStatus.textContent = 'Loading translator…'
      const choice = await createTranslatorFor(settings, {
        onProgress: (progress, text) => {
          benchStatus.textContent = `${(progress * 100).toFixed(0)}%: ${text}`
        },
      })
      if (!choice.ok) {
        benchStatus.textContent = choice.message
        return
      }
      benchTranslator = choice.translator
      benchSettings = settings
      console.log(`${TAG} bench using ${choice.label}`)
    }

    benchStatus.textContent = `Translating ${sources.length.toString()} lines…`
    const startedAt = performance.now()
    const result = await benchTranslator.translate(sources)
    const elapsed = performance.now() - startedAt

    benchBody.replaceChildren()
    for (const [index, translation] of result.translations.entries()) {
      const row = document.createElement('tr')
      for (const [text, className] of [
        [(index + 1).toString(), 'num'],
        [translation.source, 'src'],
        [translation.text, ''],
      ] as [string, string][]) {
        const cell = document.createElement('td')
        cell.className = className
        cell.textContent = text
        row.append(cell)
      }
      benchBody.append(row)
    }
    benchTable.hidden = false

    benchStatus.textContent =
      `${settings.translationProvider === 'cloud' ? 'Cloud' : 'On-device'}: ` +
      `${sources.length.toString()} lines in ${(elapsed / 1000).toFixed(1)} s ` +
      `(${(elapsed / sources.length).toFixed(0)} ms/line)`
    console.log(`${TAG} bench done in ${elapsed.toFixed(0)} ms`, result.translations)
  } catch (error) {
    console.error(`${TAG} bench failed`, error)
    benchStatus.textContent = error instanceof Error ? error.message : String(error)
  } finally {
    benchRun.disabled = false
  }
}

benchRun.addEventListener('click', () => { void runBench() })

function currentThreshold(): number {
  return Number(threshold.value) / 100
}

/** Indices into lastResult.detections that clear the current slider value. */
function visibleIndices(): number[] {
  if (!lastResult) return []
  const min = currentThreshold()
  const indices: number[] = []
  for (const [index, detection] of lastResult.detections.entries()) {
    if (detection.score >= min) indices.push(index)
  }
  return indices
}

function draw(image: ImageBitmap, detections: [number, Detection][]): void {
  canvas.width = image.width
  canvas.height = image.height

  const context = canvas.getContext('2d')
  if (!context) throw new Error('could not acquire a 2d context for the canvas')

  context.drawImage(image, 0, 0)

  // Scale annotations with the image so they stay legible on a large page.
  const weight = Math.max(2, Math.round(image.width / 500))
  context.lineWidth = weight
  context.strokeStyle = '#00e5ff'
  context.fillStyle = '#00e5ff'
  context.font = `${weight * 6}px system-ui, sans-serif`
  context.textBaseline = 'bottom'

  for (const [index, { box, score }] of detections) {
    context.strokeRect(box.x, box.y, box.width, box.height)
    context.fillText(
      `#${index} ${score.toFixed(2)}`,
      box.x,
      Math.max(weight * 6, box.y - weight),
    )
  }
}

function renderTable(indices: number[]): void {
  if (!lastResult) return

  readingsBody.replaceChildren()
  for (const index of indices) {
    const detection = lastResult.detections[index]
    if (!detection) continue
    const recognition = readings.get(index)

    const row = document.createElement('tr')
    if (recognition && recognition.confidence < WEAK_READING) row.className = 'low'

    const cells = [
      `#${index}`,
      detection.score.toFixed(2),
      recognition ? recognition.confidence.toFixed(2) : '…',
    ]
    for (const value of cells) {
      const cell = document.createElement('td')
      cell.className = 'num'
      cell.textContent = value
      row.append(cell)
    }

    const text = document.createElement('td')
    text.className = 'text'
    text.textContent = recognition ? recognition.text : ''
    row.append(text)

    const plate = document.createElement('td')
    plate.className = 'num'
    const decision = plates.get(index)
    plate.textContent = decision
      ? `${decision.toneCoverage.toFixed(2)}${decision.plated && decision.tone ? ` plate ${describeTone(decision.tone)}` : ''}`
      : ''
    if (decision?.plated) plate.classList.add('plated')
    row.append(plate)

    const english = document.createElement('td')
    english.className = 'text'
    const translation = translations.get(index)
    english.textContent = translation
      ? translation.text
      : recognition && translateToggle.checked
        ? '…'
        : ''
    row.append(english)

    readingsBody.append(row)
  }
  readingsTable.hidden = indices.length === 0
}

function render(): void {
  if (!lastImage || !lastResult) return

  const indices = visibleIndices()
  const detections = indices.map(
    (index) => [index, lastResult!.detections[index]!] as [number, Detection],
  )
  draw(lastImage, detections)
  renderTable(indices)

  const { preprocess, inference, decode } = lastResult.timings
  const pending = indices.filter((index) => !readings.has(index)).length
  status.textContent =
    `${indices.length} region${indices.length === 1 ? '' : 's'} at ≥${currentThreshold().toFixed(2)} ` +
    `· ${lastImage.width}×${lastImage.height} ` +
    `· detect: preprocess ${preprocess.toFixed(0)}ms, inference ${inference.toFixed(0)}ms, ` +
    `decode ${decode.toFixed(0)}ms` +
    (pending > 0 && ocrToggle.checked ? ` · reading ${pending.toString()}…` : '')
}

/** Read every visible region not read yet; lowering the slider reads only the new ones. */
async function ensureReadings(): Promise<void> {
  if (reading || !ocrToggle.checked || !lastImage || !lastResult) return

  const pending = visibleIndices().filter((index) => !readings.has(index))
  if (pending.length === 0) return

  const image = lastImage
  const detections = lastResult.detections
  const mine = generation
  reading = true
  try {
    const active = await getRecognizer()
    if (mine !== generation) return

    const result = await active.recognize(
      image,
      pending.map((index) => detections[index]!.box),
      (recognition, position) => {
        if (mine !== generation) return
        readings.set(pending[position]!, recognition)
        render()
      },
    )
    if (mine !== generation) return
    console.log(`${TAG} readings`, result)
  } catch (error) {
    console.error(`${TAG} OCR failed`, error)
    status.textContent = `OCR failed: ${error instanceof Error ? error.message : String(error)}`
    return
  } finally {
    reading = false
  }

  // The slider may have moved while that pass ran.
  await ensureReadings()
}

/** Erase the visible regions and draw the result beside the detections. */
async function ensureCleaned(): Promise<void> {
  if (!lastImage || !lastResult) return

  if (!cleanToggle.checked) {
    cleanedView.hidden = true
    return
  }

  const indices = visibleIndices()
  const byShape = shapeToggle.checked
  const balloonFill = balloonToggle.checked
  const keepTextOnArt = labelToggle.checked
  const key = [
    indices.join(','),
    byShape ? 'shape' : 'box',
    balloonFill ? 'balloon' : 'telea',
    keepTextOnArt ? 'label' : 'erase',
  ].join('|')
  if (cleaning || cleanedFor === key) return

  const image = lastImage
  // A detection is already an erase region; dropping its shape erases the box.
  const regions = indices.map((index) => {
    const detection = lastResult!.detections[index]!
    return byShape ? detection : { box: detection.box }
  })
  const mine = generation
  cleaning = true
  try {
    const active = await getInpainter()
    if (mine !== generation) return

    const result = await active.inpaint(image, regions, { balloonFill, keepTextOnArt })
    if (mine !== generation) {
      result.image.close()
      return
    }

    cleanedCanvas.width = result.image.width
    cleanedCanvas.height = result.image.height
    const context = cleanedCanvas.getContext('2d')
    if (!context) throw new Error('could not acquire a 2d context for the cleaned canvas')
    context.drawImage(result.image, 0, 0)
    cleanedImage?.close()
    cleanedImage = result.image
    cleanedLobes = new Map(indices.map((index, position) => [index, result.lobes[position] ?? []]))
    cleanedOnArt = new Set(indices.filter((_, position) => result.onArt[position]))
    // The typeset view reads these, so it has to follow a new erase.
    renderedFor = null

    cleanedFor = key
    cleanedView.hidden = false
    console.log(`${TAG} inpainted, ${result.filled.toString()} of ${regions.length.toString()} regions balloon-filled`, result.timings)
    // getInpainter() left "Loading OpenCV.js…" on screen; put the real status back.
    render()
  } catch (error) {
    console.error(`${TAG} inpainting failed`, error)
    status.textContent = `Inpainting failed: ${error instanceof Error ? error.message : String(error)}`
  } finally {
    cleaning = false
  }
}

/** Translate every visible region read but not yet translated. */
async function ensureTranslations(): Promise<void> {
  if (translating || !translateToggle.checked || !lastResult) return

  const pending = visibleIndices().filter(
    (index) => readings.has(index) && !translations.has(index),
  )
  if (pending.length === 0) return

  const mine = generation
  translating = true
  try {
    const active = await getTranslator()
    if (!active || mine !== generation) return

    const result = await active.translate(
      pending.map((index) => readings.get(index)!.text),
      (translation, position) => {
        if (mine !== generation) return
        translations.set(pending[position]!, translation)
        render()
      },
    )
    if (mine !== generation) return
    console.log(`${TAG} translations`, result)
  } catch (error) {
    console.error(`${TAG} translation failed`, error)
    status.textContent = `Translation failed: ${error instanceof Error ? error.message : String(error)}`
    return
  } finally {
    translating = false
  }

  await ensureTranslations()
}

/** Typeset the translations onto the erased image; needs both earlier passes. */
async function ensureRendered(): Promise<void> {
  if (rendering || !lastResult || !cleanedImage || !lastImage) return

  const indices = visibleIndices().filter((index) => translations.has(index))
  if (indices.length === 0) {
    renderedView.hidden = true
    return
  }

  const key = indices.join(',')
  if (renderedFor === key) return

  const image = cleanedImage
  const source = lastImage
  const detections = lastResult.detections
  const mine = generation
  rendering = true
  try {
    // Plates off and on over the same layout, so the two are comparable.
    renderer ??= await createCanvasRenderer({ markOverflow: true, plates: false })
    plateRenderer ??= await createCanvasRenderer()
    if (mine !== generation) return

    const inputs = indices.map((index) => ({
      box: detections[index]!.box,
      text: translations.get(index)!.text,
      lobes: orderLobes(cleanedLobes.get(index) ?? [], currentLanguage() === 'ja'),
      label: cleanedOnArt.has(index),
    }))
    const result = await renderer.render(image, source, inputs)
    const plateResult = await plateRenderer.render(image, source, inputs)
    if (mine !== generation) {
      result.image.close()
      plateResult.image.close()
      return
    }

    for (const [position, region] of plateResult.regions.entries()) {
      plates.set(indices[position]!, {
        plated: region.plated,
        toneCoverage: region.toneCoverage,
        tone: region.plateTone,
      })
    }

    for (const [target, produced] of [
      [renderedCanvas, result.image],
      [platedCanvas, plateResult.image],
    ] as const) {
      target.width = produced.width
      target.height = produced.height
      const context = target.getContext('2d')
      if (!context) throw new Error('could not acquire a 2d context for a render canvas')
      context.drawImage(produced, 0, 0)
      produced.close()
    }

    renderedFor = key
    renderedView.hidden = false
    platedView.hidden = false
    const spilled = result.regions.filter((r) => r.overflowed).length
    const plated = plateResult.regions.filter((r) => r.plated).length
    console.log(
      `${TAG} rendered`,
      { ...result.timings, overflowed: spilled, plated },
      plateResult.regions,
    )
    renderTable(indices)
  } catch (error) {
    console.error(`${TAG} rendering failed`, error)
    status.textContent = `Rendering failed: ${error instanceof Error ? error.message : String(error)}`
  } finally {
    rendering = false
  }
}

async function run(file: File): Promise<void> {
  try {
    lastImage?.close()
    lastImage = await createImageBitmap(file)
  } catch (error) {
    status.textContent = `Could not decode the image: ${error instanceof Error ? error.message : String(error)}`
    return
  }
  await detectAndProcess(lastImage)
}

/** Detection and everything after it, for a new image. */
async function detectAndProcess(image: ImageBitmap): Promise<void> {
  try {
    const active = await getDetector()

    generation++
    readings = new Map()
    translations = new Map()
    plates = new Map()
    cleanedFor = null
    cleanedImage?.close()
    cleanedImage = null
    renderedFor = null
    cleanedView.hidden = true
    renderedView.hidden = true
    platedView.hidden = true

    status.textContent = 'Detecting…'
    lastResult = await active.detect(image)
    console.log(`${TAG} detections`, lastResult)

    controls.hidden = false
    render()
    // Erasing is fast and independent of OCR, so it goes first.
    await ensureCleaned()
    await ensureReadings()
    await ensureTranslations()
    await ensureRendered()
  } catch (error) {
    console.error(`${TAG} detection failed`, error)
    status.textContent = `Failed: ${error instanceof Error ? error.message : String(error)}`
  }
}

function handleFiles(files: FileList | null | undefined): void {
  const file = files?.[0]
  if (!file) return
  if (!file.type.startsWith('image/')) {
    status.textContent = `Not an image: ${file.type || 'unknown type'}`
    return
  }
  void run(file)
}

dropZone.addEventListener('click', () => filePicker.click())
filePicker.addEventListener('change', () => handleFiles(filePicker.files))

dropZone.addEventListener('dragover', (event) => {
  event.preventDefault()
  dropZone.classList.add('over')
})
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'))
dropZone.addEventListener('drop', (event) => {
  event.preventDefault()
  dropZone.classList.remove('over')
  handleFiles(event.dataTransfer?.files)
})

document.addEventListener('paste', (event) => handleFiles(event.clipboardData?.files))

threshold.addEventListener('input', () => {
  thresholdValue.value = currentThreshold().toFixed(2)
  render()
})
threshold.addEventListener('change', () => {
  void ensureCleaned()
  void (async () => {
    await ensureReadings()
    await ensureTranslations()
    await ensureRendered()
  })()
})
for (const { value, label } of TRANSLATION_PROVIDERS) {
  providerSelect.append(new Option(label, value))
}
for (const { value, label } of TARGET_LANGUAGES) {
  targetSelect.append(new Option(label, value))
}
void loadSettings().then((settings) => {
  providerSelect.value = settings.translationProvider
  targetSelect.value = settings.targetLanguage
})
targetSelect.addEventListener('change', () => {
  void (async () => {
    await saveSettings({ targetLanguage: targetSelect.value as TargetLanguage })
    translations = new Map()
    renderedFor = null
    render()
    await ensureTranslations()
    await ensureRendered()
  })()
})
openOptions.addEventListener('click', () => void chrome.runtime.openOptionsPage())
providerSelect.addEventListener('change', () => {
  void (async () => {
    // Stored in the shared settings.
    await saveSettings({ translationProvider: providerSelect.value as TranslationProvider })
    // Translations from the other provider are discarded, not topped up.
    translations = new Map()
    renderedFor = null
    render()
    await ensureTranslations()
    await ensureRendered()
  })()
})
translateToggle.addEventListener('change', () => {
  // Translation reads the OCR output, so it is meaningless without it.
  if (translateToggle.checked && !ocrToggle.checked) ocrToggle.checked = true
  render()
  void (async () => {
    await ensureReadings()
    await ensureTranslations()
    await ensureRendered()
  })()
})
languageSelect.addEventListener('change', () => {
  // Readings came from the other model, so they and their translations are discarded.
  readings = new Map()
  translations = new Map()
  renderedFor = null
  render()
  void (async () => {
    await ensureReadings()
    await ensureTranslations()
    await ensureRendered()
  })()
})
cleanToggle.addEventListener('change', () => void ensureCleaned())
labelToggle.addEventListener('change', () => {
  renderedFor = null
  void (async () => {
    await ensureCleaned()
    await ensureRendered()
  })()
})
balloonToggle.addEventListener('change', () => {
  renderedFor = null
  void (async () => {
    await ensureCleaned()
    await ensureRendered()
  })()
})
shapeToggle.addEventListener('change', () => {
  // The render stage draws on the erased image, so it has to follow.
  renderedFor = null
  void (async () => {
    await ensureCleaned()
    await ensureRendered()
  })()
})
ocrToggle.addEventListener('change', () => {
  render()
  void ensureReadings()
})

console.log(`${TAG} ready`)
