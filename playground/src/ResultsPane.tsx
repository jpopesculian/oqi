import type { RunResult } from './runner';

export type Phase = 'booting' | 'idle' | 'running';

interface Props {
  phase: Phase;
  result: RunResult | null;
  error: string | null;
}

export function ResultsPane({ phase, result, error }: Props) {
  return (
    <div className="results-pane">
      <div className="pane-title">Results</div>
      <div className="results-body">
        {phase === 'booting' && <div className="hint">Loading wasm…</div>}
        {phase === 'running' && <div className="hint">Running…</div>}
        {phase === 'idle' && error !== null && (
          <pre className="diagnostic">{error}</pre>
        )}
        {phase === 'idle' && error === null && result !== null && (
          <Outputs result={result} />
        )}
        {phase === 'idle' && error === null && result === null && (
          <div className="hint">Press Run to execute the program.</div>
        )}
      </div>
    </div>
  );
}

function Outputs({ result }: { result: RunResult }) {
  if (result.outputs.length === 0) {
    return <div className="hint">The program produced no outputs.</div>;
  }
  return (
    <table className="outputs">
      <thead>
        <tr>
          <th>Name</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        {result.outputs.map((out) => (
          <tr key={out.name}>
            <td>{out.name}</td>
            <td className="value">{String(out.value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
