# oqi playground

A browser playground for [oqi](../): write OpenQASM 3, supply `input` values as
JSON, run it against the state-vector simulator (compiled to WebAssembly), and
see the output values — all client-side. Live at
<https://jpopesculian.github.io/oqi/>.

## Local development

The app depends on the wasm bindings in [`../js`](../js) via `file:../js/pkg`,
so that package must be built first (and rebuilt after any change under `js/`):

```sh
# from the repo root — `--features gpu` compiles in the WebGPU backend
wasm-pack build js --target web --features gpu

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
- The app **samples**: it runs the program `shots` times (default 1024) via
  `oqi-js`'s `sample()` and renders a histogram per named output variable. One RNG
  stream advances across shots, so a given seed reproduces the whole histogram.
- Inputs are a JSON object (`{ "name": value }`) passed straight to the `inputs`
  option; the seed and shots fields feed `seed` and `shots`.
- **Backend**: a toolbar dropdown selects the simulator — **CPU f64** (default),
  **CPU f32**, or **GPU** (the WebGPU `wgpu` sim, always `f32`). It maps to `sample()`'s
  `backend`/`precision` options, and the results badge echoes what ran plus the shot
  count. GPU requires a WebGPU-capable browser (Chrome/Edge, recent Firefox/Safari) over
  HTTPS — GitHub Pages qualifies; selecting GPU where it's unavailable surfaces a clear
  error. Note that GPU shots re-upload |0> and read back each measurement per shot, so
  large shot counts can be slower on GPU than CPU at small qubit sizes.

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
