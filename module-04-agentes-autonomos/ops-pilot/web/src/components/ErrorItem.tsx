import type { ErrorItem as ErrorItemModel } from "../lib/conversation.ts";
import { RequestId } from "./CopyButton.tsx";
import { Icon } from "./Icon.tsx";

const ACTION_LABELS = {
  retry: "Tentar novamente",
  open_settings: "Abrir configurações",
  new_conversation: "Nova conversa",
} as const;

interface ErrorItemProps {
  item: ErrorItemModel;
  disabled: boolean;
  onRetry: () => void;
  onOpenSettings: () => void;
  onNewConversation: () => void;
}

/** Falha numa ida ao copiloto: o que aconteceu + o que fazer, sem texto técnico (FR-006). */
export function ErrorItem({ item, disabled, onRetry, onOpenSettings, onNewConversation }: ErrorItemProps) {
  const { error } = item;
  const handlers = { retry: onRetry, open_settings: onOpenSettings, new_conversation: onNewConversation };

  return (
    <article className="callout callout-danger" aria-labelledby={`${item.id}-title`}>
      <div className="callout-title">
        <span className="status-danger">
          <Icon name="alert" />
        </span>
        <h3 id={`${item.id}-title`}>{error.title}</h3>
      </div>
      <p>{error.detail}</p>
      {error.requestId && <RequestId id={error.requestId} />}
      {error.action !== "none" && (
        <div className="callout-actions">
          {error.action === "open_settings" && (
            <button type="button" className="btn btn-secondary" onClick={onRetry} disabled={disabled}>
              {ACTION_LABELS.retry}
            </button>
          )}
          <button type="button" className="btn btn-primary" onClick={handlers[error.action]} disabled={disabled && error.action === "retry"}>
            {ACTION_LABELS[error.action]}
          </button>
        </div>
      )}
    </article>
  );
}
