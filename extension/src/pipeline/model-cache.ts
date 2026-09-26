/**
 * Measures and clears the model weights web-llm keeps in the Cache API. It sums
 * the stored responses rather than using navigator.storage.estimate(), which is
 * padded, quantised and not attributable to one cache.
 */

export interface CacheUsage {
  name: string
  entries: number
  bytes: number
  /** True when every entry's size came from a header rather than its body. */
  fromHeaders: boolean
}

async function measureCache(name: string): Promise<CacheUsage> {
  const cache = await caches.open(name)
  const requests = await cache.keys()

  let bytes = 0
  let fromHeaders = true
  for (const request of requests) {
    const response = await cache.match(request)
    if (!response) continue

    const declared = Number(response.headers.get('content-length'))
    if (Number.isFinite(declared) && declared > 0) {
      bytes += declared
    } else {
      fromHeaders = false
      bytes += (await response.blob()).size
    }
  }

  return { name, entries: requests.length, bytes, fromHeaders }
}

/**
 * Every Cache API bucket at this origin. The extension uses the Cache API only
 * for model weights, so anything here is reclaimable model data.
 */
export async function measureModelCaches(): Promise<CacheUsage[]> {
  const names = await caches.keys()
  const measured = await Promise.all(names.map(measureCache))
  return measured.sort((a, b) => b.bytes - a.bytes)
}

/** Returns the names actually deleted. */
export async function clearModelCaches(): Promise<string[]> {
  const names = await caches.keys()
  const deleted: string[] = []
  for (const name of names) {
    if (await caches.delete(name)) deleted.push(name)
  }
  return deleted
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes.toString()} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit++
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit] ?? 'GB'}`
}
