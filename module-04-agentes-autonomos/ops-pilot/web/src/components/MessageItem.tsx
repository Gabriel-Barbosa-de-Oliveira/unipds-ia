import type { ReactNode } from "react";

import type { AssistantItem, UserItem } from "../lib/conversation.ts";
import { Icon } from "./Icon.tsx";

export function UserMessage({ item }: { item: UserItem }) {
  return (
    <article className="msg msg-user" aria-label="Sua mensagem">
      <p className="bubble">{item.text}</p>
      {item.status === "failed" && (
        <p className="msg-meta msg-meta-danger">
          <Icon name="alert" />
          Não enviada
        </p>
      )}
    </article>
  );
}

/**
 * Resposta do copiloto como texto puro (`pre-wrap`): nunca interpretada como HTML (research.md
 * item 13). `actions` recebe o "Ver raciocínio" (US2).
 */
export function AssistantMessage({ item, actions }: { item: AssistantItem; actions?: ReactNode }) {
  return (
    <article className="msg msg-assistant" aria-label="Resposta do copiloto">
      <p className="bubble">{item.run.answer}</p>
      {actions && <div className="msg-actions">{actions}</div>}
    </article>
  );
}
