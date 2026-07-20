import { Histogram } from './Histogram';
import type { SampleResult } from './runner';

export type Phase = 'booting' | 'idle' | 'running';

interface Props {
  phase: Phase;
  result: SampleResult | null;
  error: string | null;
}

export function ResultsPane({ phase, result, error }: Props) {
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
          <Histograms result={result} />
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
  shots,
}: {
  backend: 'cpu' | 'gpu';
  shots: number;
}) {
  const label = `${backend === 'gpu' ? 'GPU · f32' : 'CPU · f64'} · ${shots} shots`;
  return (
    <div
      className={`backend-badge ${backend}`}
      title="Simulator backend and shot count"
    >
      {label}
    </div>
  );
}

function Histograms({ result }: { result: SampleResult }) {
  return (
    <>
      <BackendBadge backend={result.backend} shots={result.shots} />
      {result.histograms.length === 0 ? (
        <div className="hint">The program produced no outputs.</div>
      ) : (
        result.histograms.map((h) => <Histogram key={h.name} data={h} />)
      )}
    </>
  );
}
