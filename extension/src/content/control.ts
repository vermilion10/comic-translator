import { sendMessage } from '../pipeline/messages.ts'
import {
  SOURCE_LANGUAGES,
  THRESHOLD_RANGE,
  loadSettings,
  saveSettings,
} from '../pipeline/settings.ts'
import type { Box } from '../pipeline/detect/types.ts'
import {
  type PageRect,
  bestTarget,
  collectImages,
  hasPixels,
  imageAtPoint,
  isCandidate,
  isStripCandidate,
  overlapArea,
  toSourceBox,
} from './targets.ts'
import { SetupNeededError } from './setup-needed.ts'
import { findStrip, tilesTouched } from './tiles.ts'

/**
 * The floating in-page control, for pages where the context menu never
 * appears because the page cancels contextmenu. It picks the image itself (see
 * targets.ts), with hand picking and a drawn region as fallbacks.
 *
 * The UI is a closed shadow root with its layout set inline with !important
 * (page rules outrank :host), built node by node (Trusted Types forbid
 * innerHTML), and it stops every pointer event it gets from reaching the page.
 * Drawing is recorded in page coordinates so scrolling mid-drag is safe.
 */

const TAG = '[manga-translator:control]'

/** Bottom left: the other corners are usually taken by site UI or other extensions. */
const HOST_LAYOUT: [string, string][] = [
  ['position', 'fixed'],
  ['left', '16px'],
  ['bottom', '16px'],
  ['z-index', '2147483647'],
  ['margin', '0'],
  ['padding', '0'],
  ['width', 'auto'],
  ['height', 'auto'],
]

/** Pointer events during picking belong to the extension, not to the page. */
const POINTER_EVENTS = [
  'pointerdown',
  'mousedown',
  'pointerup',
  'mouseup',
  'click',
  'auxclick',
  'dblclick',
] as const

/** How long the DOM has to settle before the image inventory is rebuilt. */
const RESCAN_DELAY = 500

/** Long enough for the click that follows a pointerdown to be swallowed too. */
const PICK_TEARDOWN_DELAY = 400

/** Below this the drag reads as a misclick rather than a region. */
const MIN_DRAWN_SIDE = 6

export interface ControlHooks {
  translate: (
    image: HTMLImageElement,
    onStatus: (text: string) => void,
    box?: Box,
  ) => Promise<void>
  restore: (image: HTMLImageElement) => boolean
  isTranslated: (image: HTMLImageElement) => boolean
}

const CSS = `
:host { all: initial; }
* { box-sizing: border-box; font: 12px/1.45 system-ui, sans-serif; }
.launcher {
  width: 40px; height: 40px; border-radius: 20px; border: 0; cursor: pointer;
  background: #1c1c20; color: #fff; font-size: 14px; letter-spacing: -.5px;
  box-shadow: 0 2px 10px rgb(0 0 0 / .35);
}
.launcher:hover { background: #34343e; }
.panel {
  width: 268px; padding: 12px; border-radius: 10px; color: #f2f2f4;
  background: #1c1c20; box-shadow: 0 4px 22px rgb(0 0 0 / .45);
}
.head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.title { font-weight: 600; }
.close, .gear { background: none; border: 0; color: #9a9aa4; cursor: pointer; font-size: 16px; padding: 0 2px; }
.close:hover, .gear:hover { color: #fff; }
.tools { display: flex; align-items: center; gap: 6px; }
.target { color: #b9b9c4; margin-bottom: 8px; word-break: break-word; }
.note { color: #ffca7a; margin-bottom: 8px; }
.row { display: flex; gap: 6px; margin-bottom: 6px; }
button.act {
  flex: 1; padding: 7px 4px; border-radius: 6px; border: 1px solid #3a3a44;
  background: #2a2a32; color: #f2f2f4; cursor: pointer;
}
button.act:hover:not(:disabled) { background: #3a3a46; }
button.act:disabled { opacity: .45; cursor: default; }
button.primary { background: #3d6ae0; border-color: #3d6ae0; }
button.primary:hover:not(:disabled) { background: #5480ee; }
.region { color: #a9c7ff; margin-bottom: 8px; word-break: break-word; }
.status { color: #9fd2a0; min-height: 17px; margin: 6px 0 8px; word-break: break-word; }
.status.bad { color: #ff9d9d; }
.offer {
  border: 1px solid #4a4a58; border-radius: 6px; padding: 8px;
  margin-bottom: 8px; background: #24242c; color: #e6e6ec;
}
.offer.hidden { display: none; }
.offer p { margin: 0 0 7px; }
.settings { border-top: 1px solid #33333d; padding-top: 8px; }
.label { color: #9a9aa4; display: block; margin-bottom: 4px; }
.langs { display: flex; gap: 10px; margin-bottom: 8px; }
.langs label { display: flex; align-items: center; gap: 3px; color: #d6d6de; cursor: pointer; }
input[type=range] { width: 100%; accent-color: #3d6ae0; margin: 0; }
.hidden { display: none; }
`

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className = '',
  text = '',
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text) node.textContent = text
  return node
}

export function mountControl(hooks: ControlHooks): void {
  const host = document.createElement('manga-translator-control')
  for (const [property, value] of HOST_LAYOUT) {
    host.style.setProperty(property, value, 'important')
  }
  const shadow = host.attachShadow({ mode: 'closed' })

  const style = document.createElement('style')
  style.textContent = CSS

  const launcher = element('button', 'launcher', '文A')
  launcher.title = 'Comic Translator'

  const closeButton = element('button', 'close', '×')
  closeButton.title = 'Close'
  // Opens the settings page directly.
  const settingsButton = element('button', 'gear', '⚙')
  settingsButton.title = 'Settings'
  settingsButton.setAttribute('aria-label', 'Settings')
  const tools = element('div', 'tools')
  tools.append(settingsButton, closeButton)
  const head = element('div', 'head')
  head.append(element('span', 'title', 'Comic Translator'), tools)

  const targetLine = element('div', 'target')
  const note = element('div', 'note hidden')

  const translateButton = element('button', 'act primary', 'Translate')
  const restoreButton = element('button', 'act hidden', 'Restore')
  const topRow = element('div', 'row')
  topRow.append(translateButton, restoreButton)

  const pickButton = element('button', 'act', 'Pick image')
  const drawButton = element('button', 'act', 'Draw a box')
  drawButton.title = 'Translate one region the detector missed'
  const bottomRow = element('div', 'row')
  bottomRow.append(pickButton, drawButton)

  const regionLine = element('div', 'region hidden')
  const statusLine = element('div', 'status')

  // Shown when a run fails on something the settings page can fix.
  const offer = element('div', 'offer hidden')
  const offerText = element('p')
  const offerButton = element('button', 'act primary', 'Open settings')
  offer.append(offerText, offerButton)

  const langs = element('div', 'langs')
  const thresholdOut = document.createElement('output')
  const thresholdLabel = element('span', 'label', 'Detection threshold ')
  thresholdLabel.append(thresholdOut)
  const slider = document.createElement('input')
  slider.type = 'range'
  slider.step = '1'
  slider.min = String(Math.round(THRESHOLD_RANGE.min * 100))
  slider.max = String(Math.round(THRESHOLD_RANGE.max * 100))
  const settings = element('div', 'settings')
  settings.append(element('span', 'label', 'Source language'), langs, thresholdLabel, slider)

  const panel = element('div', 'panel hidden')
  panel.append(
    head,
    targetLine,
    note,
    topRow,
    bottomRow,
    regionLine,
    statusLine,
    offer,
    settings,
  )

  const highlight = element('div', 'highlight')
  for (const [property, value] of [
    ['position', 'fixed'],
    ['pointer-events', 'none'],
    ['border', '2px solid #3d6ae0'],
    ['background', 'rgb(61 106 224 / .12)'],
    ['display', 'none'],
  ]) {
    highlight.style.setProperty(property!, value!)
  }

  shadow.append(style, launcher, panel, highlight)

  let target: HTMLImageElement | null = null
  /** Set once the user picks by hand, so scrolling stops re-choosing for them. */
  let chosenByHand = false
  let running = false
  let picking = false
  let drawing = false
  /** A region the user drew, in source-image pixels, replacing detection. */
  let manual: { image: HTMLImageElement; box: Box; clipped: boolean; touched: number[] } | null =
    null

  // What is on screen

  const watched = new Set<HTMLImageElement>()
  const onScreen = new Set<HTMLImageElement>()

  const intersections = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const image = entry.target as HTMLImageElement
      if (entry.isIntersecting) onScreen.add(image)
      else onScreen.delete(image)
    }
    updateVisibility()
  })

  function rescan(): void {
    for (const image of collectImages()) {
      if (watched.has(image)) continue
      watched.add(image)
      intersections.observe(image)
    }
    for (const image of watched) {
      if (image.isConnected) continue
      watched.delete(image)
      onScreen.delete(image)
      intersections.unobserve(image)
    }
    updateVisibility()
  }

  /** Show the launcher only when there is something to translate; the strip test is slower, so it runs second. */
  function updateVisibility(): void {
    const open = !panel.classList.contains('hidden')
    const visible = [...onScreen]
    let worthwhile = visible.some((image) => isCandidate(image))
    if (!worthwhile) {
      const all = [...watched]
      worthwhile = visible.some((image) => isStripCandidate(image, all))
    }
    host.style.setProperty(
      'display',
      worthwhile || open || running || picking || drawing ? 'block' : 'none',
      'important',
    )
  }

  // Panel

  function describe(): void {
    describeRegion()
    if (!target) {
      targetLine.textContent = 'No image on screen is large enough to translate.'
      note.classList.add('hidden')
      translateButton.disabled = true
      restoreButton.classList.add('hidden')
      drawButton.disabled = true
      return
    }
    drawButton.disabled = running

    const rect = target.getBoundingClientRect()
    targetLine.textContent =
      `Target: ${Math.round(rect.width).toString()} × ${Math.round(rect.height).toString()} on ` +
      `screen, ${target.naturalWidth.toString()} × ${target.naturalHeight.toString()} actual.`
    translateButton.disabled = running
    translateButton.textContent = manual ? 'Translate region' : 'Translate'
    restoreButton.classList.toggle('hidden', !hooks.isTranslated(target))

    const strip = findStrip(target, [...watched])
    note.classList.toggle('hidden', !strip)
    if (strip) {
      note.textContent =
        `This page is one strip cut into ${strip.tiles.length.toString()} images, ` +
        `${Math.round(strip.height).toString()} px tall in total. Only piece ` +
        `${(strip.index + 1).toString()} gets translated, and text crossing a seam is ` +
        `cut in half. Stitching them is not built yet.`
      console.warn(
        `${TAG} tiled page: ${strip.tiles.length.toString()} images form one ` +
          `${Math.round(strip.height).toString()}px strip; ` +
          `translating piece ${(strip.index + 1).toString()} only`,
      )
    }
  }

  /**
   * What the drawn region covers. Clipping (it ran off the image) and a seam
   * crossing (the page continues into another img) are reported separately.
   */
  function describeRegion(): void {
    regionLine.classList.toggle('hidden', !manual)
    drawButton.textContent = drawing ? 'Cancel' : manual ? 'Clear box' : 'Draw a box'
    if (!manual) return

    const { box, clipped, touched } = manual
    const parts = [
      `Region: ${box.width.toString()} × ${box.height.toString()} px at ` +
        `${box.x.toString()}, ${box.y.toString()} in the image. Detection is skipped for it.`,
    ]
    if (clipped) parts.push('It ran past the edge of the image and was trimmed to fit.')
    if (touched.length > 1) {
      parts.push(
        `Only the part on this piece is translated: it also reached ` +
          `${(touched.length - 1).toString()} other piece${touched.length > 2 ? 's' : ''} of ` +
          `this strip, which the pipeline sees as separate images.`,
      )
    }
    regionLine.textContent = parts.join(' ')
  }

  function clearRegion(): void {
    manual = null
    describe()
  }

  function chooseTarget(): void {
    if (manual) return
    target = bestTarget(onScreen, [...watched])?.image ?? null
    describe()
  }

  function setStatus(text: string, bad = false): void {
    statusLine.textContent = text
    statusLine.classList.toggle('bad', bad)
  }

  function showOffer(error: SetupNeededError): void {
    offerText.textContent = error.message
    offerButton.textContent =
      error.reason === 'no-api-key' ? 'Finish setting up' : 'Set up cloud translation'
    offer.classList.remove('hidden')
  }

  function hideOffer(): void {
    offer.classList.add('hidden')
  }

  function openOptions(): void {
    void sendMessage('open-options', {}).catch((error: unknown) => {
      console.warn(`${TAG} could not open the options page`, error)
    })
  }
  offerButton.addEventListener('click', openOptions)
  settingsButton.addEventListener('click', openOptions)

  function openPanel(): void {
    launcher.classList.add('hidden')
    panel.classList.remove('hidden')
    chosenByHand = false
    manual = null
    chooseTarget()
    setStatus('')
    hideOffer()
    void fillSettings()
  }

  function closePanel(): void {
    stopPicking()
    stopDrawing()
    panel.classList.add('hidden')
    launcher.classList.remove('hidden')
    updateVisibility()
  }

  async function fillSettings(): Promise<void> {
    const current = await loadSettings()
    slider.value = String(Math.round(current.detectThreshold * 100))
    thresholdOut.textContent = current.detectThreshold.toFixed(2)

    langs.replaceChildren()
    for (const { value, label } of SOURCE_LANGUAGES) {
      const wrapper = document.createElement('label')
      const radio = document.createElement('input')
      radio.type = 'radio'
      radio.name = 'mt-source-language'
      radio.value = value
      radio.checked = value === current.sourceLanguage
      radio.addEventListener('change', () => {
        if (!radio.checked) return
        void saveSettings({ sourceLanguage: value })
      })
      wrapper.append(radio, document.createTextNode(label))
      langs.append(wrapper)
    }
  }

  slider.addEventListener('input', () => {
    thresholdOut.textContent = (Number(slider.value) / 100).toFixed(2)
  })
  slider.addEventListener('change', () => {
    // Settings are read per translation, so this applies to the next run.
    void saveSettings({ detectThreshold: Number(slider.value) / 100 })
  })

  // Actions

  async function runTranslation(): Promise<void> {
    if (!target || running) return
    running = true
    translateButton.disabled = true
    pickButton.disabled = true
    drawButton.disabled = true
    const image = target
    const box = manual?.box
    hideOffer()
    try {
      await hooks.translate(image, (text) => { setStatus(text) }, box)
      setStatus(box ? 'Region translated.' : 'Done.')
    } catch (error) {
      if (error instanceof SetupNeededError) {
        // Not styled as an error: the run stopped short of a missing setting.
        setStatus('Translation is not set up for this device.')
        showOffer(error)
      } else {
        setStatus(error instanceof Error ? error.message : String(error), true)
      }
    } finally {
      running = false
      pickButton.disabled = false
      describe()
    }
  }

  launcher.addEventListener('click', openPanel)
  closeButton.addEventListener('click', closePanel)
  translateButton.addEventListener('click', () => { void runTranslation() })
  restoreButton.addEventListener('click', () => {
    if (target && hooks.restore(target)) {
      setStatus('Original restored.')
      describe()
    }
  })
  pickButton.addEventListener('click', () => {
    if (picking) {
      stopPicking()
      setStatus('')
    } else {
      stopDrawing()
      startPicking()
    }
  })
  drawButton.addEventListener('click', () => {
    if (drawing) {
      stopDrawing()
      setStatus('')
    } else if (manual) {
      clearRegion()
      setStatus('')
    } else {
      stopPicking()
      startDrawing()
    }
  })

  // Keep the control's clicks from reaching the page, which may turn pages on click.
  for (const type of POINTER_EVENTS) {
    host.addEventListener(type, (event) => { event.stopPropagation() })
  }

  // Pointer modes

  /**
   * Both modes take pointer events from the page. On window in the capture
   * phase, so they also pre-empt listeners the page put on document.
   */
  function bindPointers(
    on: boolean,
    handlers: { pointer: (event: Event) => void; move: (event: Event) => void; key: (event: Event) => void },
  ): void {
    const bind = on
      ? window.addEventListener.bind(window)
      : window.removeEventListener.bind(window)
    for (const type of POINTER_EVENTS) bind(type, handlers.pointer, true)
    bind('pointermove', handlers.move, true)
    bind('keydown', handlers.key, true)
  }

  function showHighlight(rect: PageRect): void {
    highlight.style.display = 'block'
    highlight.style.left = `${(rect.left - window.scrollX).toString()}px`
    highlight.style.top = `${(rect.top - window.scrollY).toString()}px`
    highlight.style.width = `${rect.width.toString()}px`
    highlight.style.height = `${rect.height.toString()}px`
  }

  // Picking an image by hand

  let highlighted: HTMLImageElement | null = null

  function startPicking(): void {
    if (picking) return
    picking = true
    setStatus('Click the image to translate. Esc cancels.')
    pickButton.textContent = 'Cancel'
    bindPointers(true, { pointer: onPickPointer, move: onPickMove, key: onPickKey })
  }

  function stopPicking(): void {
    if (!picking) return
    picking = false
    pickButton.textContent = 'Pick image'
    highlight.style.display = 'none'
    highlighted = null
    bindPointers(false, { pointer: onPickPointer, move: onPickMove, key: onPickKey })
  }

  function onPickMove(event: Event): void {
    const pointer = event as PointerEvent
    const found = imageAtPoint(pointer.clientX, pointer.clientY, host)
    if (found === highlighted) return
    highlighted = found
    if (!found) {
      highlight.style.display = 'none'
      return
    }
    const rect = found.getBoundingClientRect()
    showHighlight({
      left: rect.left + window.scrollX,
      top: rect.top + window.scrollY,
      width: rect.width,
      height: rect.height,
    })
  }

  function onPickPointer(event: Event): void {
    // The control's own buttons still have to work while picking.
    if (event.composedPath().includes(host)) return

    event.preventDefault()
    event.stopImmediatePropagation()
    if (event.type !== 'pointerdown') return

    const pointer = event as PointerEvent
    const found = imageAtPoint(pointer.clientX, pointer.clientY, host)
    if (!found) {
      setStatus('No image under the pointer there.', true)
    } else if (!hasPixels(found)) {
      // Lazy viewers stretch a 1x1 placeholder over the tile, so this is common.
      setStatus('That image has not loaded yet. Scroll it into view and try again.', true)
    } else {
      target = found
      chosenByHand = true
      describe()
      setStatus('Image selected.')
    }
    // Tear down after the click that follows, or the page still receives it.
    window.setTimeout(stopPicking, PICK_TEARDOWN_DELAY)
  }

  function onPickKey(event: Event): void {
    if ((event as KeyboardEvent).key !== 'Escape') return
    event.preventDefault()
    event.stopImmediatePropagation()
    stopPicking()
    setStatus('')
  }

  // Drawing a region by hand

  /** Drag origin in page coordinates, so scrolling mid-drag does not smear it. */
  let dragStart: { x: number; y: number } | null = null

  function startDrawing(): void {
    if (drawing) return
    drawing = true
    manual = null
    dragStart = null
    setStatus('Drag a box around the text. Esc cancels.')
    describeRegion()
    bindPointers(true, { pointer: onDrawPointer, move: onDrawMove, key: onDrawKey })
  }

  function stopDrawing(): void {
    if (!drawing) return
    drawing = false
    dragStart = null
    highlight.style.display = 'none'
    bindPointers(false, { pointer: onDrawPointer, move: onDrawMove, key: onDrawKey })
    describeRegion()
  }

  function dragRect(x: number, y: number): PageRect | null {
    if (!dragStart) return null
    const left = Math.min(dragStart.x, x)
    const top = Math.min(dragStart.y, y)
    return { left, top, width: Math.abs(x - dragStart.x), height: Math.abs(y - dragStart.y) }
  }

  function onDrawMove(event: Event): void {
    const pointer = event as PointerEvent
    const rect = dragRect(pointer.clientX + window.scrollX, pointer.clientY + window.scrollY)
    if (rect) showHighlight(rect)
  }

  /** The image holding most of a drawn rectangle, not the one under the drag's start point. */
  function ownerOf(rect: PageRect): HTMLImageElement | null {
    let best: HTMLImageElement | null = null
    let bestArea = 0
    for (const image of watched) {
      if (!hasPixels(image)) continue
      const area = overlapArea(image, rect)
      if (area > bestArea) {
        bestArea = area
        best = image
      }
    }
    return bestArea > 0 ? best : null
  }

  function finishDrawing(rect: PageRect): void {
    if (rect.width < MIN_DRAWN_SIDE || rect.height < MIN_DRAWN_SIDE) {
      setStatus('That box was too small. Drag across the text you want translated.', true)
      return
    }

    const owner = ownerOf(rect)
    if (!owner) {
      setStatus('That box does not land on an image.', true)
      return
    }

    const mapped = toSourceBox(owner, rect)
    if (!mapped) {
      setStatus('That box does not land on an image.', true)
      return
    }

    const strip = findStrip(owner, [...watched])
    target = owner
    chosenByHand = true
    manual = {
      image: owner,
      box: mapped.box,
      clipped: mapped.clipped,
      touched: strip ? tilesTouched(strip, rect) : [],
    }
    describe()
    setStatus('Region selected. Press Translate region.')
    if (manual.touched.length > 1) {
      console.warn(
        `${TAG} drawn region spans ${manual.touched.length.toString()} tiles of a strip; ` +
          `only the part on the target image is translated`,
      )
    }
  }

  function onDrawPointer(event: Event): void {
    // The control's own buttons still have to work while drawing.
    if (event.composedPath().includes(host)) return

    event.preventDefault()
    event.stopImmediatePropagation()

    const pointer = event as PointerEvent
    const x = pointer.clientX + window.scrollX
    const y = pointer.clientY + window.scrollY

    if (event.type === 'pointerdown') {
      dragStart = { x, y }
      showHighlight({ left: x, top: y, width: 0, height: 0 })
      return
    }
    if (event.type !== 'pointerup' || !dragStart) return

    const rect = dragRect(x, y)
    dragStart = null
    if (rect) finishDrawing(rect)
    // Tear down after the click that follows, or the page still receives it.
    window.setTimeout(stopDrawing, PICK_TEARDOWN_DELAY)
  }

  function onDrawKey(event: Event): void {
    if ((event as KeyboardEvent).key !== 'Escape') return
    event.preventDefault()
    event.stopImmediatePropagation()
    stopDrawing()
    setStatus('')
  }

  // Lifecycle

  document.documentElement.append(host)

  let rescanTimer = 0
  const mutations = new MutationObserver(() => {
    window.clearTimeout(rescanTimer)
    rescanTimer = window.setTimeout(rescan, RESCAN_DELAY)
  })
  mutations.observe(document.documentElement, { childList: true, subtree: true })

  window.addEventListener(
    'scroll',
    () => {
      if (chosenByHand || running || picking || drawing) return
      if (!panel.classList.contains('hidden')) chooseTarget()
    },
    { passive: true },
  )

  rescan()
  console.log(`${TAG} floating control mounted`)
}
