import js from '@eslint/js'
import globals from 'globals'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  {
    // public/ holds fetched weights and vendored library code, none of it ours.
    ignores: ['dist/**', 'public/**', '.crxjs/**'],
  },
  js.configs.recommended,
  tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
      globals: {
        ...globals.browser,
        ...globals.webextensions,
      },
    },
  },
  {
    // Build config and scripts run in Node, not the browser.
    files: ['*.config.ts', 'eslint.config.js', 'scripts/**/*.mjs'],
    languageOptions: {
      globals: globals.node,
    },
  },
  {
    // These live outside the type-checked program, so type-aware rules cannot
    // run against them.
    files: ['eslint.config.js', 'scripts/**/*.mjs'],
    extends: [tseslint.configs.disableTypeChecked],
  },
)
