// Main-thread client for the wasm worker: spawns it, matches responses to
// requests by id, and supports Stop via terminate + respawn.

export type InputValue = boolean | number | string | InputValue[];

export type BackendChoice = 'cpu' | 'gpu' | 'auto';
export type Precision = 'f32' | 'f64';

export interface RunOptions {
  inputs: Record<string, InputValue>;
  seed?: number | bigint;
  backend?: BackendChoice;
  precision?: Precision;
  shots: number;
}

export interface HistoBar {
  label: string;
  count: number;
}

export interface Histogram {
  name: string;
  total: number;
  bars: HistoBar[];
}

/** Byte-offset range into the source for an error's offending span. */
export interface ErrorSpan {
  start: number;
  end: number;
}

export interface SampleResult {
  shots: number;
  /** The backend that actually ran (`auto` reports `cpu` or `gpu`). */
  backend: 'cpu' | 'gpu';
  /** The amplitude precision that ran. */
  precision: Precision;
  /** One histogram per named output variable, in program order. */
  histograms: Histogram[];
}

type WorkerReply =
  | { type: 'ready' }
  | { type: 'result'; id: number; ok: true; value: SampleResult }
  | {
      type: 'result';
      id: number;
      ok: false;
      error: string;
      span: ErrorSpan | null;
    };

export class RunnerStoppedError extends Error {
  constructor() {
    super('stopped');
    this.name = 'RunnerStoppedError';
  }
}

/** A failed run: the rendered diagnostic message plus its source span, if any. */
export class RunError extends Error {
  span: ErrorSpan | null;
  constructor(message: string, span: ErrorSpan | null) {
    super(message);
    this.name = 'RunError';
    this.span = span;
  }
}

interface Pending {
  resolve: (result: SampleResult) => void;
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
        entry.reject(new RunError(msg.error, msg.span));
      }
    };
  }

  /** Sample the program over `options.shots` and return per-variable histograms. */
  run(source: string, options: RunOptions): Promise<SampleResult> {
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
