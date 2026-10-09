import { useEffect, useRef } from "react";

import type { Decision } from "../lib/approval-machine.ts";
import type { ChatRun, ConversationState } from "../lib/conversation.ts";
import { ApprovalCard } from "./ApprovalCard.tsx";
import { EmptyState } from "./EmptyState.tsx";
import { ErrorItem } from "./ErrorItem.tsx";
import { Icon } from "./Icon.tsx";
import { AssistantMessage, UserMessage } from "./MessageItem.tsx";
import { ThinkingIndicator } from "./ThinkingIndicator.tsx";

interface ConversationLogProps {
  state: ConversationState;
  onPickExample: (text: string) => void;
  onShowReasoning: (run: ChatRun) => void;
  onRetry: (errorItemId: string) => void;
  onOpenSettings: () => void;
  onNewConversation: () => void;
  onDecide: (itemId: string, approvalId: string, decision: Decision) => void;
}

/** A conversa (`role=log`): novas respostas e cartões são anunciados a leitores de tela. */
export function ConversationLog(props: ConversationLogProps) {
  const { state } = props;
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    endRef.current?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "end" });
  }, [state.items.length, state.pending]);

  const reasoningButton = (run: ChatRun) => (
    <button type="button" className="btn btn-ghost" aria-haspopup="dialog" onClick={() => props.onShowReasoning(run)}>
      <Icon name="reasoning" />
      Ver raciocínio
    </button>
  );

  return (
    <main className="conversation">
      <div className="conversation-inner" role="log" aria-live="polite" aria-relevant="additions" aria-label="Conversa com o copiloto">
        {state.items.length === 0 && <EmptyState onPick={props.onPickExample} disabled={state.pending !== "idle"} />}
        {state.items.map((item) => {
          switch (item.kind) {
            case "user":
              return <UserMessage key={item.id} item={item} />;
            case "assistant":
              return <AssistantMessage key={item.id} item={item} actions={reasoningButton(item.run)} />;
            case "error":
              return (
                <ErrorItem
                  key={item.id}
                  item={item}
                  disabled={state.pending !== "idle"}
                  onRetry={() => props.onRetry(item.id)}
                  onOpenSettings={props.onOpenSettings}
                  onNewConversation={props.onNewConversation}
                />
              );
            case "approval":
              return (
                <ApprovalCard
                  key={item.id}
                  item={item}
                  onDecide={(decision) => props.onDecide(item.id, item.approval.id, decision)}
                  onShowReasoning={() => props.onShowReasoning(item.run)}
                />
              );
          }
        })}
        {state.pending === "sending" && <ThinkingIndicator />}
        <div ref={endRef} />
      </div>
    </main>
  );
}
