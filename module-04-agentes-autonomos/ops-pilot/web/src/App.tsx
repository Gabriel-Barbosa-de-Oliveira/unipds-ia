import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { postChat, postDecision } from "./api/client.ts";
import { loadSettings, saveSettings } from "./api/storage.ts";
import type { Decision } from "./lib/approval-machine.ts";
import { chatReducer, initialConversation, type ChatRun, type ConversationState } from "./lib/conversation.ts";
import { resolveTheme, type Settings } from "./lib/settings.ts";
import { Composer } from "./components/Composer.tsx";
import { ConversationLog } from "./components/ConversationLog.tsx";
import { Header } from "./components/Header.tsx";
import { ReasoningPanel } from "./components/ReasoningPanel.tsx";
import { SettingsPanel } from "./components/SettingsPanel.tsx";

const makeId = () => crypto.randomUUID();

const DARK_QUERY = "(prefers-color-scheme: dark)";

/** Aplica o tema e acompanha o sistema quando a preferência é "system" (FR-024). */
function useTheme(preference: Settings["theme"]) {
  useEffect(() => {
    const media = window.matchMedia(DARK_QUERY);
    const apply = () => document.documentElement.setAttribute("data-theme", resolveTheme(preference, media.matches));
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, [preference]);
}

export function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [conversation, dispatch] = useReducer(chatReducer, initialConversation);
  const [openRun, setOpenRun] = useState<ChatRun | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // O reducer decide se o envio é aceito; o efeito de rede lê o estado mais recente por ref.
  const stateRef = useRef<ConversationState>(conversation);
  stateRef.current = conversation;

  useTheme(settings.theme);

  const runChat = useCallback(
    async (message: string, conversationId: string | null) => {
      const result = await postChat(settings.apiUrl, conversationId ? { message, conversationId } : { message });
      if (result.kind === "error") {
        dispatch({ type: "failed", id: makeId(), error: result.error });
      } else {
        dispatch({ type: "received", id: makeId(), result: result.data });
      }
    },
    [settings.apiUrl],
  );

  const send = useCallback(
    (text: string): boolean => {
      const before = stateRef.current;
      const action = { type: "send", id: makeId(), text } as const;
      const next = chatReducer(before, action);
      if (next === before) {
        return false;
      }
      // Adianta a ref: um segundo clique antes do re-render já vê `sending` (sem envio duplicado).
      stateRef.current = next;
      dispatch(action);
      void runChat(text.trim(), before.conversationId);
      return true;
    },
    [runChat],
  );

  const retry = useCallback(
    (errorItemId: string) => {
      const before = stateRef.current;
      const target = before.items.find((item) => item.kind === "error" && item.id === errorItemId);
      if (!target || target.kind !== "error" || before.pending !== "idle") {
        return;
      }
      const action = { type: "retry", errorItemId, id: makeId() } as const;
      stateRef.current = chatReducer(before, action);
      dispatch(action);
      void runChat(target.retryText, before.conversationId);
    },
    [runChat],
  );

  const decide = useCallback(
    async (itemId: string, approvalId: string, decision: Decision) => {
      const item = stateRef.current.items.find((candidate) => candidate.id === itemId);
      if (!item || item.kind !== "approval" || item.card.status !== "pending") {
        return;
      }
      const action = { type: "approvalUpdated", itemId, action: { type: "decide", decision } } as const;
      stateRef.current = chatReducer(stateRef.current, action);
      dispatch(action);
      const result = await postDecision(settings.apiUrl, approvalId, decision);
      if (result.kind === "error") {
        dispatch({ type: "approvalUpdated", itemId, action: { type: "failed", error: result.error } });
        return;
      }
      dispatch({ type: "approvalUpdated", itemId, action: { type: "succeeded", status: result.data.approval.status } });
      dispatch({ type: "decisionReceived", itemId, id: makeId(), data: result.data });
    },
    [settings.apiUrl],
  );

  const changeSettings = useCallback((next: Settings) => {
    saveSettings(next);
    setSettings(next);
  }, []);

  const newConversation = useCallback(() => {
    setOpenRun(null);
    dispatch({ type: "reset" });
  }, []);

  const disabledReason =
    conversation.pending === "awaiting_decision"
      ? "Decida a ação pendente antes de enviar outra mensagem."
      : conversation.pending === "sending"
        ? "Aguarde a resposta do copiloto."
        : undefined;

  return (
    <div className="app">
      <Header
        onNewConversation={newConversation}
        onOpenSettings={() => setSettingsOpen(true)}
        canReset={conversation.items.length > 0 && conversation.pending !== "sending"}
      />
      <ConversationLog
        state={conversation}
        onPickExample={send}
        onShowReasoning={setOpenRun}
        onRetry={retry}
        onOpenSettings={() => setSettingsOpen(true)}
        onNewConversation={newConversation}
        onDecide={decide}
      />
      <Composer onSend={send} disabled={conversation.pending !== "idle"} disabledReason={disabledReason} />
      {openRun && <ReasoningPanel run={openRun} onClose={() => setOpenRun(null)} />}
      {settingsOpen && <SettingsPanel settings={settings} onChange={changeSettings} onClose={() => setSettingsOpen(false)} />}
    </div>
  );
}
