// Web Worker that owns the wasm module. `run()` is CPU-synchronous inside
// wasm, so it lives here to keep the UI responsive; the main thread can
// terminate() a runaway program.
import init, { run } from 'oqi-js';

import type { RunOptions } from './runner';

interface RunRequest {
  type: 'run';
  id: number;
  source: string;
  options: RunOptions;
}

const ready = init().then(() => {
  self.postMessage({ type: 'ready' });
});

self.onmessage = async (event: MessageEvent<RunRequest>) => {
  const msg = event.data;
  if (msg.type !== 'run') return;
  await ready;
  try {
    const value = await run(msg.source, msg.options);
    self.postMessage({ type: 'result', id: msg.id, ok: true, value });
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: 'result', id: msg.id, ok: false, error });
  }
};
