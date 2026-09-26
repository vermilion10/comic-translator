/**
 * A failure the user can fix on the options page (no WebGPU, no API key, a
 * cloud-only target language). The control offers the fix instead of printing
 * a message; the reason crosses the port as a string and is rebuilt here.
 */
import type { SetupReason } from '../pipeline/translate/for-provider.ts'

export class SetupNeededError extends Error {
  constructor(
    readonly reason: SetupReason,
    message: string,
  ) {
    super(message)
    this.name = 'SetupNeededError'
  }
}
