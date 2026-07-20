import type { BackendSel } from './App';
import { EXAMPLES } from './examples';
import type { Phase } from './ResultsPane';

const BACKEND_OPTIONS: { value: BackendSel; label: string }[] = [
  { value: 'cpu-f64', label: 'CPU f64' },
  { value: 'cpu-f32', label: 'CPU f32' },
  { value: 'gpu', label: 'GPU' },
];

interface Props {
  phase: Phase;
  seed: string;
  seedError: string | null;
  onSeedChange: (seed: string) => void;
  shots: string;
  shotsError: string | null;
  onShotsChange: (shots: string) => void;
  backend: BackendSel;
  onBackendChange: (backend: BackendSel) => void;
  onRun: () => void;
  onStop: () => void;
  onLoadExample: (index: number) => void;
}

export function Toolbar({
  phase,
  seed,
  seedError,
  onSeedChange,
  shots,
  shotsError,
  onShotsChange,
  backend,
  onBackendChange,
  onRun,
  onStop,
  onLoadExample,
}: Props) {
  return (
    <header className="toolbar">
      <span className="brand">oqi playground</span>
      {phase === 'running' ? (
        <button className="run stop" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button className="run" onClick={onRun} disabled={phase !== 'idle'}>
          Run
        </button>
      )}
      <label className="field">
        shots
        <input
          className="num"
          value={shots}
          onChange={(e) => onShotsChange(e.target.value)}
          inputMode="numeric"
          spellCheck={false}
        />
      </label>
      {shotsError !== null && <span className="inline-error">{shotsError}</span>}
      <label className="field">
        seed
        <input
          className="num"
          value={seed}
          onChange={(e) => onSeedChange(e.target.value)}
          placeholder="default"
          spellCheck={false}
        />
      </label>
      {seedError !== null && <span className="inline-error">{seedError}</span>}
      <label className="field">
        backend
        <select
          className="backend-select"
          value={backend}
          onChange={(e) => onBackendChange(e.target.value as BackendSel)}
        >
          {BACKEND_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </label>
      <span className="spacer" />
      <select
        className="examples"
        value=""
        onChange={(e) => {
          const index = Number(e.target.value);
          if (!Number.isNaN(index) && e.target.value !== '') {
            onLoadExample(index);
          }
        }}
      >
        <option value="" disabled>
          Load example…
        </option>
        {EXAMPLES.map((ex, i) => (
          <option key={ex.name} value={i}>
            {ex.name}
          </option>
        ))}
      </select>
    </header>
  );
}
