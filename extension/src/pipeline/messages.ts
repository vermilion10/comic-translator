/**
 * The message contract between content scripts and the background worker.
 * Every exchange is declared in MessageMap.
 */
export interface MessageMap {
  /** Sent once per frame when a content script finishes booting. */
  'content-script-ready': {
    request: { url: string }
    response: { receivedAt: number }
  }
  /** Background -> content script: the context menu was used on the image at srcUrl. */
  'translate-image': {
    request: { srcUrl: string }
    response: { accepted: boolean }
  }
  /** Background -> content script, to put a translated image back as it was. */
  'restore-image': {
    request: { srcUrl: string }
    response: { restored: boolean }
  }
  /**
   * Content script -> background, to open the settings page, which a content
   * script cannot open itself. Reuses an already-open settings tab.
   */
  'open-options': {
    request: Record<string, never>
    response: { opened: boolean }
  }
  /**
   * Content script -> background: make sure the shared pipeline document exists
   * before connecting a 'pipeline' port to it. Only the worker can create it.
   */
  'ensure-pipeline': {
    request: Record<string, never>
    response: { ready: boolean }
  }
  /** Pipeline document -> background, once its port listener is registered. */
  'pipeline-ready': {
    request: Record<string, never>
    response: { acknowledged: boolean }
  }
}

export type MessageType = keyof MessageMap

export type RequestFor<T extends MessageType> = MessageMap[T]['request']

export type ResponseFor<T extends MessageType> = MessageMap[T]['response']

/** A request as it travels over chrome.runtime; `type` discriminates the union. */
export type Message<T extends MessageType = MessageType> = {
  [K in MessageType]: { type: K; payload: RequestFor<K> }
}[T]

/** Type-safe wrapper over chrome.runtime.sendMessage, for content -> background. */
export function sendMessage<T extends MessageType>(
  type: T,
  payload: RequestFor<T>,
): Promise<ResponseFor<T>> {
  // TypeScript cannot relate the generic payload to the mapped union.
  const message = { type, payload } as Message
  return chrome.runtime.sendMessage(message)
}

/** Type-safe wrapper over chrome.tabs.sendMessage, for background -> content. */
export function sendToTab<T extends MessageType>(
  tabId: number,
  type: T,
  payload: RequestFor<T>,
): Promise<ResponseFor<T>> {
  const message = { type, payload } as Message
  return chrome.tabs.sendMessage(tabId, message)
}
