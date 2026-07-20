import { EXAMPLES } from './examples';
import type { Phase } from './ResultsPane';

interface Props {
  phase: Phase;
  seed: string;
  seedError: string | null;
  onSeedChange: (seed: string) => void;
  shots: string;
  shotsError: string | null;
  onShotsChange: (shots: string) => void;
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
