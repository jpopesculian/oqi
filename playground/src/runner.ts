// Main-thread client for the wasm worker: spawns it, matches responses to
// requests by id, and supports Stop via terminate + respawn.

export type InputValue = boolean | number | string;

export type BackendChoice = 'cpu' | 'gpu' | 'auto';

export interface RunOptions {
  inputs: Record<string, InputValue>;
  seed?: number | bigint;
  backend?: BackendChoice;
}

export interface RunResult {
  outputs: { name: string; value: InputValue }[];
  measurements: { qubit: number; value: boolean }[];
  /** The backend that actually ran (`auto` reports `cpu` or `gpu`). */
  backend: 'cpu' | 'gpu';
}

type WorkerReply =
  | { type: 'ready' }
  | { type: 'result'; id: number; ok: true; value: RunResult }
  | { type: 'result'; id: number; ok: false; error: string };

export class RunnerStoppedError extends Error {
  constructor() {
    super('stopped');
    this.name = 'RunnerStoppedError';
  }
}

interface Pending {
  resolve: (result: RunResult) => void;
  reject: (error: Error) => void;
}

export class Runner {
  private worker!: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();

  /** Called with `true` once the (re)spawned worker's wasm is initialized. */
  onReadyChange: (ready: boolean) => void = () => {};

  constructor() {
    this.spawn();
  }

  private spawn() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), {
      type: 'module',
    });
    this.worker.onmessage = (event: MessageEvent<WorkerReply>) => {
      const msg = event.data;
      if (msg.type === 'ready') {
        this.onReadyChange(true);
        return;
      }
      const entry = this.pending.get(msg.id);
      if (!entry) return; // stale reply from a superseded run
      this.pending.delete(msg.id);
      if (msg.ok) {
        entry.resolve(msg.value);
      } else {
        entry.reject(new Error(msg.error));
      }
    };
  }

  run(source: string, options: RunOptions): Promise<RunResult> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: 'run', id, source, options });
    });
  }

  /** Kill a runaway run: terminate the worker and boot a fresh one. */
  stop() {
    this.worker.terminate();
    for (const entry of this.pending.values()) {
      entry.reject(new RunnerStoppedError());
    }
    this.pending.clear();
    this.onReadyChange(false);
    this.spawn();
  }

  dispose() {
    this.worker.terminate();
  }
}
