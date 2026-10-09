import type { TraceView } from "../lib/trace-view.ts";
import { Icon } from "./Icon.tsx";

function CodeBody({ text, long }: { text: string; long: boolean }) {
  const block = <pre className="code-block">{text}</pre>;
  return long ? (
    <details>
      <summary>Mostrar conteúdo ({text.length.toLocaleString("pt-BR")} caracteres)</summary>
      {block}
    </details>
  ) : (
    block
  );
}

function Body({ view }: { view: TraceView }) {
  const { body } = view;
  switch (body.kind) {
    case "text":
      return <p className="trace-body">{body.text}</p>;
    case "steps":
      return (
        <ol className="trace-steps">
          {body.steps.map((step, index) => (
            <li key={index}>{step}</li>
          ))}
        </ol>
      );
    case "code":
      return (
        <>
          {body.title && (
            <p>
              Ferramenta <code>{body.title}</code>
            </p>
          )}
          <CodeBody text={body.text} long={body.long} />
        </>
      );
    case "route":
      return (
        <p className="trace-body">
          <code>{body.route}</code> — {body.reason} <span className="hint">({body.source})</span>
        </p>
      );
    case "fallback":
      return (
        <p className="trace-body">
          <code>{body.from}</code> → <code>{body.to}</code> — {body.reason}
        </p>
      );
    case "handoff":
      return (
        <>
          <p>
            <strong>
              {body.from} → {body.to}
            </strong>
          </p>
          <p className="trace-body">{body.brief}</p>
        </>
      );
  }
}

/** Um evento do trace: ícone + rótulo do tipo, etapa do fluxo e conteúdo próprio (FR-009/010). */
export function TraceEventItem({ view, position }: { view: TraceView; position: number }) {
  return (
    <li className={`trace-event trace-${view.kind}`}>
      <div className="trace-head">
        <span className="trace-label">
          <Icon name={view.icon} />
          {view.label}
        </span>
        {view.node && (
          <span className="badge" title="Etapa do fluxo">
            {view.node}
          </span>
        )}
        {view.role && (
          <span className="badge" title="Papel da equipe">
            {view.role}
          </span>
        )}
        <span className="trace-index">#{position}</span>
      </div>
      <Body view={view} />
    </li>
  );
}
