/**
 * Messages between a content script and the pipeline document, over a
 * chrome.runtime port named PIPELINE_PORT, one port per translation. The result
 * travels as a PNG data URL because a port serialises to JSON.
 */

export const PIPELINE_PORT = 'pipeline'

import type { Box } from '../../pipeline/detect/types.ts'
import type { Settings } from '../../pipeline/settings.ts'
import type { SetupReason } from '../../pipeline/translate/for-provider.ts'

export type RunnerRequest = {
  type: 'runner-translate'
  id: number
  /**
   * The original image URL. The pipeline fetches it, since a content script's
   * fetch is bound by the page's CORS rules. Never a blob: URL of an earlier
   * result; the pipeline keys its retained render on this.
   */
  srcUrl: string
  /**
   * One region in source-image pixels, used instead of detection: the manual
   * box for text the detector missed.
   */
  box?: Box
  /** Read by the content script; an offscreen document has no chrome.storage. */
  settings: Settings
}

export type RunnerResponse =
  | { type: 'runner-progress'; id: number; detail: string }
  | {
      type: 'runner-result'
      id: number
      /** The translated page as a PNG data URL. */
      image: string
      regions: number
    }
  | {
      type: 'runner-error'
      id: number
      message: string
      /** Set when the run failed on something the user can fix in settings. */
      reason?: SetupReason
    }
