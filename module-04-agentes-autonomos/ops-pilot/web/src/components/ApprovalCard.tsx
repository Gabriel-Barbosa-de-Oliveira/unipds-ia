import type { Decision } from "../lib/approval-machine.ts";
import type { ApprovalItem } from "../lib/conversation.ts";
import { prettyValue } from "../lib/trace-view.ts";
import { Icon, type IconName } from "./Icon.tsx";

interface ApprovalCardProps {
  item: ApprovalItem;
  onDecide: (decision: Decision) => void;
  onShowReasoning: () => void;
}

interface StatusView {
  icon: IconName;
  text: string;
  tone: "warning" | "success" | "danger" | "muted";
}

function statusView(item: ApprovalItem): StatusView {
  const { card } = item;
  switch (card.status) {
    case "pending":
      return { icon: "warning", text: "Aguardando", tone: "warning" };
    case "submitting":
      return { icon: "clock", text: card.decision === "approve" ? "Aprovando…" : "Negando…", tone: "muted" };
    case "approved":
      return { icon: "check", text: "Aprovada", tone: "success" };
    case "denied":
      return { icon: "cross", text: "Negada", tone: "danger" };
    case "unavailable":
      return { icon: "alert", text: card.reason, tone: "muted" };
  }
}

const CALLOUT_TONE = { warning: "callout-warning", success: "callout-success", danger: "callout-muted", muted: "callout-muted" };

function formatTime(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
}

/** Ação que muda a produção aguardando decisão humana (US3, contracts/web-ui.md). */
export function ApprovalCard({ item, onDecide, onShowReasoning }: ApprovalCardProps) {
  const { approval, card } = item;
  const status = statusView(item);
  const titleId = `${item.id}-title`;
  const enabled = card.status === "pending";

  return (
    <article role="group" className={`callout ${CALLOUT_TONE[status.tone]}`} aria-labelledby={titleId}>
      <div className="callout-title">
        <span className="status-warning">
          <Icon name="warning" />
        </span>
        <h3 id={titleId}>Ação aguardando aprovação</h3>
        <span className={`callout-status status-${status.tone}`}>
          <Icon name={status.icon} />
          {status.text}
        </span>
      </div>

      <p>
        <strong>{approval.summary}</strong>
      </p>
      {approval.reason && <p>Motivo: {approval.reason}</p>}
      {card.status === "pending" && <p className="hint">Expira às {formatTime(approval.expiresAt)}. Nada foi executado ainda.</p>}

      <details>
        <summary>
          Detalhes (<code>{approval.tool}</code>)
        </summary>
        <pre className="code-block">{prettyValue(approval.args)}</pre>
      </details>

      {card.status === "pending" && card.error && (
        <p className="inline-error" role="alert">
          {card.error.title}. {card.error.detail}
        </p>
      )}

      <div className="callout-actions">
        <button type="button" className="btn btn-ghost" onClick={onShowReasoning} aria-haspopup="dialog">
          <Icon name="reasoning" />
          Ver raciocínio
        </button>
        <button type="button" className="btn btn-secondary" onClick={() => onDecide("deny")} disabled={!enabled}>
          Negar
        </button>
        <button type="button" className="btn btn-primary" onClick={() => onDecide("approve")} disabled={!enabled}>
          Aprovar
        </button>
      </div>
    </article>
  );
}
