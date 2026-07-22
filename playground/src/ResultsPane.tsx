import { Histogram } from './Histogram';
import type { SampleResult } from './runner';

export type Phase = 'booting' | 'idle' | 'running';

interface Props {
  phase: Phase;
  result: SampleResult | null;
  error: string | null;
  elapsedMs: number | null;
}

/** Human-readable run duration, e.g. `240ms`, `1.5s`, `30s`. */
function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = ms / 1000;
  return s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`;
}

export function ResultsPane({ phase, result, error, elapsedMs }: Props) {
  return (
    <div className="results-pane">
      <div className="pane-title">Results</div>
      <div className="results-body">
        {phase === 'booting' && <div className="hint">Loading wasm…</div>}
        {phase === 'running' && <div className="hint">Sampling…</div>}
        {phase === 'idle' && error !== null && (
          <pre className="diagnostic">{error}</pre>
        )}
        {phase === 'idle' && error === null && result !== null && (
          <Histograms result={result} elapsedMs={elapsedMs} />
        )}
        {phase === 'idle' && error === null && result === null && (
          <div className="hint">Press Run to sample the program.</div>
        )}
      </div>
    </div>
  );
}

function BackendBadge({
  backend,
  precision,
  shots,
}: {
  backend: 'cpu' | 'gpu';
  precision: 'f32' | 'f64';
  shots: number;
}) {
  const label = `${backend === 'gpu' ? 'GPU' : 'CPU'} · ${precision} · ${shots} shots`;
  return (
    <div
      className={`backend-badge ${backend}`}
      title="Simulator backend, precision, and shot count"
    >
      {label}
    </div>
  );
}

function Histograms({
  result,
  elapsedMs,
}: {
  result: SampleResult;
  elapsedMs: number | null;
}) {
  return (
    <>
      <div className="run-summary">
        <BackendBadge
          backend={result.backend}
          precision={result.precision}
          shots={result.shots}
        />
        {elapsedMs !== null && (
          <span className="run-time">{formatDuration(elapsedMs)}</span>
        )}
      </div>
      {result.histograms.length === 0 ? (
        <div className="hint">The program produced no outputs.</div>
      ) : (
        result.histograms.map((h) => <Histogram key={h.name} data={h} />)
      )}
    </>
  );
}
