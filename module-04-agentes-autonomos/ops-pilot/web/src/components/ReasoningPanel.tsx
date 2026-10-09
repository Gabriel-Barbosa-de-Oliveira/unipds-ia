import type { ChatRun } from "../lib/conversation.ts";
import { summarizeRun, toTraceView } from "../lib/trace-view.ts";
import { RequestId } from "./CopyButton.tsx";
import { Dialog } from "./Dialog.tsx";
import { TraceEventItem } from "./TraceEventItem.tsx";

/** "Ver raciocínio": rota, métricas, id e o trace tipado da execução (US2). */
export function ReasoningPanel({ run, onClose }: { run: ChatRun; onClose: () => void }) {
  const summary = summarizeRun(run);

  return (
    <Dialog title="Raciocínio" onClose={onClose}>
      <section className="run-summary" aria-labelledby="run-summary-title">
        <h3 id="run-summary-title">Resumo da execução</h3>
        {summary.route && (
          <p>
            <span className="badge">Rota: {summary.route.route}</span> {summary.route.reason}{" "}
            <span className="hint">({summary.route.source})</span>
          </p>
        )}
        {summary.metrics && (
          <dl className="metrics">
            {summary.metrics.map((metric) => (
              <div key={metric.label}>
                <dt>{metric.label}</dt>
                <dd>{metric.value}</dd>
              </div>
            ))}
          </dl>
        )}
        <RequestId id={summary.requestId} />
      </section>

      <section className="run-summary" aria-labelledby="trace-title">
        <h3 id="trace-title">Eventos ({run.trace.length})</h3>
        {run.trace.length === 0 ? (
          <p className="hint">Essa execução não registrou eventos.</p>
        ) : (
          <ol className="trace-list">
            {run.trace.map((event, index) => (
              <TraceEventItem key={index} view={toTraceView(event)} position={index + 1} />
            ))}
          </ol>
        )}
      </section>
    </Dialog>
  );
}
