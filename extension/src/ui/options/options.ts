import {
  type CacheUsage,
  clearModelCaches,
  formatBytes,
  measureModelCaches,
} from '../../pipeline/model-cache.ts'
import {
  SOURCE_LANGUAGES,
  type SourceLanguage,
  THRESHOLD_RANGE,
  TRANSLATION_PROVIDERS,
  type TranslationProvider,
  loadSettings,
  saveSettings,
  TARGET_LANGUAGES,
  type TargetLanguage,
} from '../../pipeline/settings.ts'
import { DEFAULT_GEMINI_MODEL } from '../../pipeline/translate/gemini-translator.ts'
import { detectWebGpu } from '../../pipeline/translate/webgpu.ts'

/**
 * The settings page. Choosing Cloud shows a callout, not a hint: it sends the
 * recognised text to a third party, and a privacy change the user does not
 * notice is one they did not agree to.
 */

const TAG = '[manga-translator:options]'

const threshold = requireElement<HTMLInputElement>('threshold')
const thresholdValue = requireElement<HTMLOutputElement>('threshold-value')
const language = requireElement<HTMLElement>('language')
const languageNote = requireElement<HTMLElement>('language-note')
const target = requireElement<HTMLElement>('target')
const provider = requireElement<HTMLElement>('provider')
const webgpuLine = requireElement<HTMLElement>('webgpu')
const cloudSetup = requireElement<HTMLElement>('cloud-setup')
const key = requireElement<HTMLInputElement>('key')
const keyClear = requireElement<HTMLButtonElement>('key-clear')
const keyStatus = requireElement<HTMLElement>('key-status')
const cachesBody = requireElement<HTMLTableSectionElement>('caches-body')
const clearButton = requireElement<HTMLButtonElement>('clear')
const status = requireElement<HTMLElement>('status')

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id)
  if (!element) throw new Error(`options markup is missing #${id}`)
  return element as T
}

/** Which model each choice loads, so the size is visible before choosing. */
const LANGUAGE_NOTE: Record<SourceLanguage, string> = {
  ja: 'Japanese uses manga-ocr, which reads the vertical text manga is set in.',
  zh:
    'Chinese uses PP-OCRv6, which reads the horizontal text manhua is set in. '
    + 'Vertical Chinese is read poorly by every model here.',
  ko:
    'Korean uses PaddleOCR’s Korean model. Neither of the other two can write '
    + 'hangul at all, so this has to be set rather than guessed.',
}

function renderLanguage(selected: SourceLanguage): void {
  language.replaceChildren()
  for (const { value, label } of SOURCE_LANGUAGES) {
    const wrapper = document.createElement('label')
    const radio = document.createElement('input')
    radio.type = 'radio'
    radio.name = 'source-language'
    radio.value = value
    radio.checked = value === selected
    radio.addEventListener('change', () => {
      if (!radio.checked) return
      languageNote.textContent = LANGUAGE_NOTE[value]
      void (async () => {
        await saveSettings({ sourceLanguage: value })
        console.log(`${TAG} saved source language ${value}`)
      })()
    })
    wrapper.append(radio, document.createTextNode(label))
    language.append(wrapper)
  }
  languageNote.textContent = LANGUAGE_NOTE[selected]
}

function renderTarget(selected: TargetLanguage): void {
  target.replaceChildren()
  for (const { value, label } of TARGET_LANGUAGES) {
    const wrapper = document.createElement('label')
    const radio = document.createElement('input')
    radio.type = 'radio'
    radio.name = 'target-language'
    radio.value = value
    radio.checked = value === selected
    radio.addEventListener('change', () => {
      if (!radio.checked) return
      void (async () => {
        await saveSettings({ targetLanguage: value })
        console.log(`${TAG} saved target language ${value}`)
      })()
    })
    wrapper.append(radio, document.createTextNode(label))
    target.append(wrapper)
  }
}

/** The key field appears only when Cloud is selected. */
function renderProvider(selected: TranslationProvider): void {
  provider.replaceChildren()
  for (const { value, label } of TRANSLATION_PROVIDERS) {
    const wrapper = document.createElement('label')
    const radio = document.createElement('input')
    radio.type = 'radio'
    radio.name = 'translation-provider'
    radio.value = value
    radio.checked = value === selected
    radio.addEventListener('change', () => {
      if (!radio.checked) return
      cloudSetup.hidden = value !== 'cloud'
      void (async () => {
        await saveSettings({ translationProvider: value })
        console.log(`${TAG} saved translation provider ${value}`)
      })()
    })
    wrapper.append(radio, document.createTextNode(label))
    provider.append(wrapper)
  }
  cloudSetup.hidden = selected !== 'cloud'
}

/** Say whether the local stage can run here, with the same check the runner uses. */
async function renderWebGpu(): Promise<void> {
  const support = await detectWebGpu()
  webgpuLine.classList.toggle('missing', !support.available)
  webgpuLine.textContent = support.available
    ? 'This device supports WebGPU, so on-device translation works here.'
    : `On-device translation cannot run here: ${support.detail}. `
      + 'Cloud translation is the way to translate on this device — read what it '
      + 'means below before turning it on.'
}

/** Never echoes the key itself. */
function renderKeyStatus(saved: string): void {
  keyClear.disabled = saved.length === 0
  keyStatus.textContent = saved
    ? `Key saved (${saved.length.toString()} characters). Using ${DEFAULT_GEMINI_MODEL}.`
    : 'No key saved yet.'
}

/** The slider works in whole percent; the setting is a 0..1 fraction. */
function sliderToThreshold(value: string): number {
  return Number(value) / 100
}

function renderCaches(usage: CacheUsage[]): void {
  cachesBody.replaceChildren()

  if (usage.length === 0) {
    const row = document.createElement('tr')
    const cell = document.createElement('td')
    cell.colSpan = 3
    cell.className = 'empty'
    cell.textContent = 'Nothing cached yet. The model downloads on first translation.'
    row.append(cell)
    cachesBody.append(row)
    clearButton.disabled = true
    return
  }

  for (const entry of usage) {
    const row = document.createElement('tr')
    const cells = [
      { text: entry.name, className: '' },
      { text: entry.entries.toString(), className: 'num' },
      { text: formatBytes(entry.bytes), className: 'num' },
    ]
    for (const { text, className } of cells) {
      const cell = document.createElement('td')
      cell.className = className
      cell.textContent = text
      row.append(cell)
    }
    cachesBody.append(row)
  }

  const total = usage.reduce((sum, entry) => sum + entry.bytes, 0)
  const items = usage.reduce((sum, entry) => sum + entry.entries, 0)
  const totalRow = document.createElement('tr')
  totalRow.className = 'total'
  for (const [text, className] of [
    ['Total', ''],
    [items.toString(), 'num'],
    [formatBytes(total), 'num'],
  ]) {
    const cell = document.createElement('td')
    cell.className = className ?? ''
    cell.textContent = text ?? ''
    totalRow.append(cell)
  }
  cachesBody.append(totalRow)
  clearButton.disabled = false
}

/** Returns a note about the measurement, or '' when there is nothing to say. */
async function refreshCaches(): Promise<string> {
  status.textContent = 'Measuring…'
  try {
    const usage = await measureModelCaches()
    renderCaches(usage)
    // A size read from a body means the whole entry was loaded to weigh it.
    const note = usage.some((entry) => !entry.fromHeaders)
      ? 'Some sizes were measured by reading the stored data.'
      : ''
    status.textContent = note
    return note
  } catch (error) {
    console.error(`${TAG} could not measure caches`, error)
    status.textContent = 'Could not read the cache.'
    return ''
  }
}

threshold.addEventListener('input', () => {
  thresholdValue.value = sliderToThreshold(threshold.value).toFixed(2)
})
threshold.addEventListener('change', () => {
  void (async () => {
    await saveSettings({ detectThreshold: sliderToThreshold(threshold.value) })
    console.log(`${TAG} saved threshold ${threshold.value}%`)
  })()
})

// 'change', not 'input', so partial keys are not saved on every keystroke.
key.addEventListener('change', () => {
  void (async () => {
    const value = key.value.trim()
    await saveSettings({ cloudApiKey: value })
    // Cleared once stored, so the key does not end up in a screenshot.
    key.value = ''
    renderKeyStatus(value)
    console.log(`${TAG} ${value ? 'saved' : 'cleared'} the cloud API key`)
  })()
})

keyClear.addEventListener('click', () => {
  void (async () => {
    await saveSettings({ cloudApiKey: '' })
    key.value = ''
    renderKeyStatus('')
    console.log(`${TAG} cleared the cloud API key`)
  })()
})

clearButton.addEventListener('click', () => {
  void (async () => {
    clearButton.disabled = true
    status.textContent = 'Clearing…'
    try {
      const deleted = await clearModelCaches()
      // Refresh first: refreshCaches owns the status line and would wipe the message.
      await refreshCaches()
      status.textContent =
        deleted.length > 0
          ? `Cleared ${deleted.length.toString()} cache${deleted.length === 1 ? '' : 's'}.`
          : 'Nothing to clear.'
    } catch (error) {
      console.error(`${TAG} could not clear caches`, error)
      status.textContent = 'Could not clear the cache.'
      clearButton.disabled = false
    }
  })()
})

void (async () => {
  const settings = await loadSettings()
  renderLanguage(settings.sourceLanguage)
  renderTarget(settings.targetLanguage)
  renderProvider(settings.translationProvider)
  renderKeyStatus(settings.cloudApiKey)
  // Not awaited: requestAdapter can be slow on a cold GPU.
  void renderWebGpu()
  threshold.min = (THRESHOLD_RANGE.min * 100).toString()
  threshold.max = (THRESHOLD_RANGE.max * 100).toString()
  threshold.value = Math.round(settings.detectThreshold * 100).toString()
  thresholdValue.value = settings.detectThreshold.toFixed(2)
  await refreshCaches()
})()
