import type { Histogram as HistogramData } from './runner';

/** A horizontal bar chart of observed values for one output variable. */
export function Histogram({ data }: { data: HistogramData }) {
  const max = data.bars.reduce((m, b) => Math.max(m, b.count), 1);
  return (
    <div className="histogram">
      <div className="histogram-title">{data.name}</div>
      <div className="bars">
        {data.bars.map((bar) => {
          const pct = (bar.count / data.total) * 100;
          return (
            <div className="bar-row" key={bar.label}>
              <span className="bar-label">{bar.label}</span>
              <span className="bar-track">
                <span
                  className="bar-fill"
                  style={{ width: `${(bar.count / max) * 100}%` }}
                />
              </span>
              <span className="bar-count">
                {bar.count}
                <span className="bar-pct"> {pct.toFixed(1)}%</span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
