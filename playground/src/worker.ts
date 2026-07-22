// Web Worker that owns the wasm module. Sampling is CPU-synchronous inside
// wasm, so it lives here to keep the UI responsive; the main thread can
// terminate() a runaway run.
import init, { sample } from 'oqi-js';

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
    const value = await sample(msg.source, msg.options);
    self.postMessage({ type: 'result', id: msg.id, ok: true, value });
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    // Diagnostic errors carry a `{ start, end }` byte-offset span for the
    // offending source; plain errors don't.
    const span =
      err && typeof err === 'object' && 'span' in err
        ? (err as { span: unknown }).span
        : null;
    self.postMessage({ type: 'result', id: msg.id, ok: false, error, span });
  }
};
