import { type Message, type ResponseFor, sendToTab } from '../pipeline/messages'

const TAG = '[manga-translator:background]'

const TRANSLATE_ID = 'manga-translator-translate'
const RESTORE_ID = 'manga-translator-restore'

console.log(`${TAG} service worker started`)

/** The pipeline document; see src/ui/runner/runner.ts. */
const PIPELINE_PAGE = 'src/ui/runner/index.html'

/** Resolves when the pipeline document reports its port listener is up. */
let pipelineReady: Promise<void> | null = null
let markPipelineReady: (() => void) | null = null

/**
 * Create the shared pipeline document if there is none, and wait until it is
 * listening. One per browser session, so loaded models serve every tab.
 */
async function ensurePipeline(): Promise<void> {
  if (await chrome.offscreen.hasDocument()) {
    // Created by an earlier worker lifetime, or still starting in this one.
    if (pipelineReady) await pipelineReady
    return
  }
  pipelineReady = new Promise<void>((resolve) => {
    markPipelineReady = resolve
  })
  await chrome.offscreen.createDocument({
    url: PIPELINE_PAGE,
    // It hosts the OpenCV sandbox iframe, and onnxruntime's threads are workers.
    reasons: [chrome.offscreen.Reason.IFRAME_SCRIPTING, chrome.offscreen.Reason.WORKERS],
    justification: 'Runs the translation models once for every tab.',
  })
  await pipelineReady
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: TRANSLATE_ID,
      title: 'Translate this comic page',
      contexts: ['image'],
    })
    chrome.contextMenus.create({
      id: RESTORE_ID,
      title: 'Restore original image',
      contexts: ['image'],
    })
  })
})

chrome.contextMenus.onClicked.addListener((info, tab) => {
  const tabId = tab?.id
  if (tabId === undefined || info.srcUrl === undefined) return
  if (info.menuItemId !== TRANSLATE_ID && info.menuItemId !== RESTORE_ID) return

  const type = info.menuItemId === TRANSLATE_ID ? 'translate-image' : 'restore-image'
  // The content script owns the DOM; this only says which image was clicked.
  sendToTab(tabId, type, { srcUrl: info.srcUrl }).catch((error: unknown) => {
    console.error(`${TAG} could not reach the content script`, error)
  })
})

chrome.runtime.onMessage.addListener((message: Message, sender, sendResponse) => {
  switch (message.type) {
    case 'content-script-ready': {
      const tab = sender.tab?.id ?? 'unknown'
      console.log(`${TAG} content script ready in tab ${tab}: ${message.payload.url}`)

      const response: ResponseFor<'content-script-ready'> = { receivedAt: Date.now() }
      sendResponse(response)
      break
    }
    case 'open-options': {
      chrome.runtime.openOptionsPage(() => {
        if (chrome.runtime.lastError) {
          console.error(`${TAG} could not open the options page`, chrome.runtime.lastError)
        }
      })
      const opened: ResponseFor<'open-options'> = { opened: true }
      sendResponse(opened)
      break
    }
    case 'ensure-pipeline': {
      ensurePipeline().then(
        () => {
          const response: ResponseFor<'ensure-pipeline'> = { ready: true }
          sendResponse(response)
        },
        (error: unknown) => {
          console.error(`${TAG} could not start the pipeline document`, error)
          const response: ResponseFor<'ensure-pipeline'> = { ready: false }
          sendResponse(response)
        },
      )
      // Answered asynchronously, so the channel has to stay open.
      return true
    }
    case 'pipeline-ready': {
      markPipelineReady?.()
      const response: ResponseFor<'pipeline-ready'> = { acknowledged: true }
      sendResponse(response)
      break
    }
    case 'translate-image':
    case 'restore-image':
      // Sent by this worker to a tab, never received here.
      break
    default:
      // Other extensions and page scripts can post here too; ignore unknown types.
      console.warn(`${TAG} ignoring unrecognised message`, message)
  }

  // Every other handler above answers synchronously, so the channel can close.
  return false
})
