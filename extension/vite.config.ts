import { fileURLToPath } from 'node:url'

import { crx } from '@crxjs/vite-plugin'
import { defineConfig } from 'vite'

import manifest from './manifest.config.ts'

export default defineConfig({
  plugins: [crx({ manifest })],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      // The runner (opened as an offscreen document) and the harness are reached
      // only by URL, so CRXJS does not discover them as HTML entries.
      input: {
        runner: fileURLToPath(new URL('src/ui/runner/index.html', import.meta.url)),
        harness: fileURLToPath(new URL('src/ui/harness/index.html', import.meta.url)),
      },
      output: {
        // onnxruntime-web gets its own chunk: its thread workers load the script it
        // lives in, and inside a page chunk they crash on `document is not defined`.
        manualChunks: (id) => (id.includes('onnxruntime-web') ? 'onnxruntime' : undefined),
      },
    },
  },
  // CRXJS serves HMR on a fixed port the extension cannot follow elsewhere.
  server: {
    port: 5173,
    strictPort: true,
    hmr: { port: 5173 },
  },
})
