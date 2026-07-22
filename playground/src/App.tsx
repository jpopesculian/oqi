import { useEffect, useRef, useState } from 'react';

import { EXAMPLES } from './examples';
import { InputsPane } from './InputsPane';
import { ResultsPane, type Phase } from './ResultsPane';
import {
  RunError,
  Runner,
  RunnerStoppedError,
  type ErrorSpan,
  type InputValue,
  type SampleResult,
} from './runner';
import { SourceEditor } from './SourceEditor';
import { Toolbar } from './Toolbar';

const MAX_SEED = 1n << 64n;
const MAX_SHOTS = 100_000;

/** Backend dropdown choices, mapped to oqi-js `{ backend, precision }`. */
export type BackendSel = 'cpu-f64' | 'cpu-f32' | 'gpu';

const BACKENDS: Record<
  BackendSel,
  { label: string; backend: 'cpu' | 'gpu'; precision: 'f32' | 'f64' }
> = {
  'cpu-f64': { label: 'CPU · f64', backend: 'cpu', precision: 'f64' },
  'cpu-f32': { label: 'CPU · f32', backend: 'cpu', precision: 'f32' },
  gpu: { label: 'GPU · f32', backend: 'gpu', precision: 'f32' },
};

/** Positive integer shot count, clamped by the wasm side to 1..=100_000. */
function parseShots(text: string): number {
  const trimmed = text.trim();
  if (!/^\d+$/.test(trimmed)) {
    throw new Error('shots must be a positive integer');
  }
  const n = Number(trimmed);
  if (n < 1) throw new Error('shots must be at least 1');
  if (n > MAX_SHOTS) throw new Error(`shots must be ≤ ${MAX_SHOTS}`);
  return n;
}

/** A scalar (bool/number/string) or an array of valid inputs (for `array[…]`). */
function isValidInput(value: unknown): value is InputValue {
  const kind = typeof value;
  if (kind === 'boolean' || kind === 'number' || kind === 'string') return true;
  return Array.isArray(value) && value.every(isValidInput);
}

/** Blank → `{}`; otherwise a JSON object of scalar or array input values. */
function parseInputs(text: string): Record<string, InputValue> {
  const trimmed = text.trim();
  if (trimmed === '') return {};
  const parsed: unknown = JSON.parse(trimmed);
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('inputs must be a JSON object');
  }
  for (const [key, value] of Object.entries(parsed)) {
    if (!isValidInput(value)) {
      throw new Error(
        `inputs.${key} must be a boolean, number, string, or array of them`,
      );
    }
  }
  return parsed as Record<string, InputValue>;
}

/** Blank → default seed; otherwise a u64, as number when safely integral. */
function parseSeed(text: string): number | bigint | undefined {
  const trimmed = text.trim();
  if (trimmed === '') return undefined;
  if (!/^\d+$/.test(trimmed)) {
    throw new Error('seed must be a non-negative integer');
  }
  const value = BigInt(trimmed);
  if (value >= MAX_SEED) {
    throw new Error('seed must fit in 64 bits');
  }
  return value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value;
}

export function App() {
  const [source, setSource] = useState(EXAMPLES[0].source);
  const [inputsText, setInputsText] = useState(EXAMPLES[0].inputs);
  const [seed, setSeed] = useState(EXAMPLES[0].seed);
  const [shots, setShots] = useState('1024');
  const [backendSel, setBackendSel] = useState<BackendSel>('cpu-f32');
  const [phase, setPhase] = useState<Phase>('booting');
  const [result, setResult] = useState<SampleResult | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [errorSpan, setErrorSpan] = useState<ErrorSpan | null>(null);
  const [inputsError, setInputsError] = useState<string | null>(null);
  const [seedError, setSeedError] = useState<string | null>(null);
  const [shotsError, setShotsError] = useState<string | null>(null);

  const runnerRef = useRef<Runner | null>(null);
  const handleRunRef = useRef<() => void>(() => {});

  useEffect(() => {
    const runner = new Runner();
    runner.onReadyChange = (ready) => {
      setPhase(ready ? 'idle' : 'booting');
    };
    runnerRef.current = runner;
    return () => {
      runnerRef.current = null;
      runner.dispose();
    };
  }, []);

  // Ctrl/Cmd+Enter runs, from anywhere on the page (incl. the editor).
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        handleRunRef.current();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const handleRun = () => {
    const runner = runnerRef.current;
    if (!runner || phase !== 'idle') return;

    setInputsError(null);
    setSeedError(null);
    setShotsError(null);
    setRunError(null);
    setErrorSpan(null);

    let inputs: Record<string, InputValue>;
    try {
      inputs = parseInputs(inputsText);
    } catch (err) {
      setInputsError(err instanceof Error ? err.message : String(err));
      return;
    }
    let seedValue: number | bigint | undefined;
    try {
      seedValue = parseSeed(seed);
    } catch (err) {
      setSeedError(err instanceof Error ? err.message : String(err));
      return;
    }
    let shotCount: number;
    try {
      shotCount = parseShots(shots);
    } catch (err) {
      setShotsError(err instanceof Error ? err.message : String(err));
      return;
    }

    setPhase('running');
    setResult(null);
    setElapsedMs(null);
    const chosen = BACKENDS[backendSel];
    const start = performance.now();
    runner
      .run(source, {
        inputs,
        seed: seedValue,
        backend: chosen.backend,
        precision: chosen.precision,
        shots: shotCount,
      })
      .then(
      (res) => {
        setResult(res);
        setElapsedMs(performance.now() - start);
        setPhase('idle');
      },
      (err: unknown) => {
        if (err instanceof RunnerStoppedError) {
          // The worker is respawning; onReadyChange restores the phase.
          setRunError('Run stopped.');
          return;
        }
        setRunError(err instanceof Error ? err.message : String(err));
        setErrorSpan(err instanceof RunError ? err.span : null);
        setPhase('idle');
      },
    );
  };

  handleRunRef.current = handleRun;

  const handleStop = () => {
    runnerRef.current?.stop();
  };

  const handleLoadExample = (index: number) => {
    const example = EXAMPLES[index];
    if (!example) return;
    setSource(example.source);
    setInputsText(example.inputs);
    setSeed(example.seed);
    setResult(null);
    setElapsedMs(null);
    setRunError(null);
    setErrorSpan(null);
    setInputsError(null);
    setSeedError(null);
    setShotsError(null);
  };

  return (
    <div className="app">
      <Toolbar
        phase={phase}
        seed={seed}
        seedError={seedError}
        onSeedChange={setSeed}
        shots={shots}
        shotsError={shotsError}
        onShotsChange={setShots}
        backend={backendSel}
        onBackendChange={setBackendSel}
        onRun={handleRun}
        onStop={handleStop}
        onLoadExample={handleLoadExample}
      />
      <div className="panes">
        <div className="left">
          <div className="source-pane">
            <div className="pane-title">OpenQASM</div>
            <SourceEditor
              value={source}
              onChange={setSource}
              errorSpan={errorSpan}
            />
          </div>
          <InputsPane
            value={inputsText}
            onChange={setInputsText}
            error={inputsError}
          />
        </div>
        <ResultsPane
          phase={phase}
          result={result}
          error={runError}
          elapsedMs={elapsedMs}
        />
      </div>
    </div>
  );
}
