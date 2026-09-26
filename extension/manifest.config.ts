import { defineManifest } from '@crxjs/vite-plugin'

import pkg from './package.json' with { type: 'json' }

export default defineManifest({
  manifest_version: 3,
  name: 'Comic Translator',
  version: pkg.version,
  description: pkg.description,
  background: {
    service_worker: 'src/background/service-worker.ts',
    type: 'module',
  },
  permissions: ['contextMenus', 'storage', 'offscreen'],
  // Lets the pipeline fetch images from any host: image CDNs often send no CORS
  // headers, and only an extension-origin document with host permission is exempt.
  host_permissions: ['<all_urls>'],
  // Cross-origin isolation for extension pages, which gives onnxruntime threads.
  // CRXJS's manifest type does not list these two keys; it passes them through.
  ...({
    cross_origin_embedder_policy: { value: 'require-corp' },
    cross_origin_opener_policy: { value: 'same-origin' },
  } as Record<string, unknown>),
  content_scripts: [
    {
      matches: ['<all_urls>'],
      js: ['src/content/content-script.ts'],
      run_at: 'document_idle',
    },
  ],
  options_ui: {
    page: 'src/ui/options/index.html',
    open_in_tab: true,
  },
  // OpenCV.js compiles code with `new Function`, which only a sandboxed page may do.
  sandbox: {
    pages: ['src/ui/sandbox/inpaint.html'],
  },
  content_security_policy: {
    // onnxruntime-web compiles its WASM at runtime.
    extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'",
    sandbox:
      "sandbox allow-scripts; script-src 'self' 'unsafe-eval' 'wasm-unsafe-eval'; object-src 'self'",
  },
})
