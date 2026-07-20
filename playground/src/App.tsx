import { useEffect, useRef, useState } from 'react';

import { EXAMPLES } from './examples';
import { InputsPane } from './InputsPane';
import { ResultsPane, type Phase } from './ResultsPane';
import {
  Runner,
  RunnerStoppedError,
  type InputValue,
  type SampleResult,
} from './runner';
import { SourceEditor } from './SourceEditor';
import { Toolbar } from './Toolbar';

const MAX_SEED = 1n << 64n;
const MAX_SHOTS = 100_000;

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

/** Blank → `{}`; otherwise a JSON object of boolean|number|string values. */
function parseInputs(text: string): Record<string, InputValue> {
  const trimmed = text.trim();
  if (trimmed === '') return {};
  const parsed: unknown = JSON.parse(trimmed);
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('inputs must be a JSON object');
  }
  for (const [key, value] of Object.entries(parsed)) {
    const kind = typeof value;
    if (kind !== 'boolean' && kind !== 'number' && kind !== 'string') {
      throw new Error(`inputs.${key} must be a boolean, number, or string`);
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
  const [phase, setPhase] = useState<Phase>('booting');
  const [result, setResult] = useState<SampleResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [inputsError, setInputsError] = useState<string | null>(null);
  const [seedError, setSeedError] = useState<string | null>(null);
  const [shotsError, setShotsError] = useState<string | null>(null);

  const runnerRef = useRef<Runner | null>(null);

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

  const handleRun = () => {
    const runner = runnerRef.current;
    if (!runner || phase !== 'idle') return;

    setInputsError(null);
    setSeedError(null);
    setShotsError(null);
    setRunError(null);

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
    // Prefer WebGPU (f32) when the browser exposes it; the wasm side falls back
    // to the f64 CPU sim and reports which backend actually ran.
    runner
      .run(source, {
        inputs,
        seed: seedValue,
        backend: 'auto',
        shots: shotCount,
      })
      .then(
      (res) => {
        setResult(res);
        setPhase('idle');
      },
      (err: unknown) => {
        if (err instanceof RunnerStoppedError) {
          // The worker is respawning; onReadyChange restores the phase.
          setRunError('Run stopped.');
          return;
        }
        setRunError(err instanceof Error ? err.message : String(err));
        setPhase('idle');
      },
    );
  };

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
    setRunError(null);
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
        onRun={handleRun}
        onStop={handleStop}
        onLoadExample={handleLoadExample}
      />
      <div className="panes">
        <div className="left">
          <div className="source-pane">
            <div className="pane-title">OpenQASM</div>
            <SourceEditor value={source} onChange={setSource} />
          </div>
          <InputsPane
            value={inputsText}
            onChange={setInputsText}
            error={inputsError}
          />
        </div>
        <ResultsPane phase={phase} result={result} error={runError} />
      </div>
    </div>
  );
}
