# oqi playground

A browser playground for [oqi](../): write OpenQASM 3, supply `input` values as
JSON, run it against the state-vector simulator (compiled to WebAssembly), and
see the output values — all client-side. Live at
<https://jpopesculian.github.io/oqi/>.

## Local development

The app depends on the wasm bindings in [`../js`](../js) via `file:../js/pkg`,
so that package must be built first (and rebuilt after any change under `js/`):

```sh
# from the repo root
wasm-pack build js --target web

cd playground
npm install
npm run dev        # http://localhost:5173/oqi/  (note the /oqi/ base path)
```

Production build / preview:

```sh
npm run build      # tsc --noEmit && vite build → dist/
npm run preview    # http://localhost:4173/oqi/
```

## How it works

- **`src/worker.ts`** owns the wasm module and calls `oqi-js`'s `run()`. Because
  `run()` is CPU-synchronous inside wasm, it lives in a Web Worker so the UI stays
  responsive and a runaway program can be terminated with **Stop**.
- **`src/runner.ts`** is the main-thread client: it matches worker replies to
  requests by id and re-spawns the worker on Stop.
- **`src/qasm.ts`** is a small CodeMirror `StreamLanguage` highlighter for
  OpenQASM (keyword lists mirror the lexer in `../lex`).
- Inputs are a JSON object (`{ "name": value }`) passed straight to `run()`'s
  `inputs` option; the seed field feeds its `seed` option for reproducible runs.

## Notes

- `js/pkg` is gitignored and built by CI, so the deployed site always tracks the
  current bindings.
- If the wasm asset ever fails to resolve (e.g. a bundler that rewrites
  `import.meta.url`), import it explicitly:
  `import wasmUrl from 'oqi-js/oqi_js_bg.wasm?url'` then `init({ module_or_path: wasmUrl })`.

## Deployment

Pushing to `main` runs [`.github/workflows/pages.yml`](../.github/workflows/pages.yml),
which builds the wasm package and the site and publishes `playground/dist` to
GitHub Pages. One-time setup: in the repo's **Settings → Pages**, set
**Source = "GitHub Actions"**.
