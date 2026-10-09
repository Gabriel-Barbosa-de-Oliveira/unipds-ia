import type {
  Approval,
  ChatAwaitingApproval,
  ChatOk,
  DecisionOk,
  Metrics,
  RouteDecision,
  TraceEvent,
} from "./api-schemas.ts";
import {
  approvalReducer,
  initialApprovalCard,
  isAwaitingDecision,
  type ApprovalCardAction,
  type ApprovalCardState,
} from "./approval-machine.ts";
import type { UiError } from "./errors.ts";

/** Uma ida ao copiloto — o que o "ver raciocínio" abre (data-model.md, ChatRun). */
export interface ChatRun {
  requestId: string;
  answer: string | null;
  trace: TraceEvent[];
  route: RouteDecision | null;
  metrics: Metrics | null;
}

export type UserItem = { kind: "user"; id: string; text: string; status: "sending" | "delivered" | "failed" };
export type AssistantItem = { kind: "assistant"; id: string; run: ChatRun };
export type ErrorItem = { kind: "error"; id: string; error: UiError; retryText: string; userItemId: string };
export type ApprovalItem = { kind: "approval"; id: string; approval: Approval; run: ChatRun; card: ApprovalCardState };

export type ConversationItem = UserItem | AssistantItem | ErrorItem | ApprovalItem;

export type PendingState = "idle" | "sending" | "awaiting_decision";

export interface ConversationState {
  conversationId: string | null;
  items: ConversationItem[];
  pending: PendingState;
}

export const initialConversation: ConversationState = { conversationId: null, items: [], pending: "idle" };

/** Ids vêm de fora (`makeId` no componente) para o reducer continuar puro. */
export type ConversationAction =
  | { type: "send"; id: string; text: string }
  | { type: "received"; id: string; result: ChatOk | ChatAwaitingApproval }
  | { type: "failed"; id: string; error: UiError }
  | { type: "retry"; errorItemId: string; id: string }
  | { type: "reset" }
  | { type: "approvalUpdated"; itemId: string; action: ApprovalCardAction }
  | { type: "decisionReceived"; itemId: string; id: string; data: DecisionOk };

function derivePending(items: readonly ConversationItem[], sending: boolean): PendingState {
  if (sending) {
    return "sending";
  }
  return items.some((item) => item.kind === "approval" && isAwaitingDecision(item.card)) ? "awaiting_decision" : "idle";
}

function withItems(state: ConversationState, items: ConversationItem[], sending: boolean): ConversationState {
  return { ...state, items, pending: derivePending(items, sending) };
}

function settleSendingUser(items: readonly ConversationItem[], status: "delivered" | "failed"): ConversationItem[] {
  return items.map((item) => (item.kind === "user" && item.status === "sending" ? { ...item, status } : item));
}

function runOf(result: ChatOk | ChatAwaitingApproval | DecisionOk): ChatRun {
  return {
    requestId: result.requestId,
    answer: "answer" in result ? result.answer : null,
    trace: result.trace,
    route: result.route,
    metrics: result.metrics,
  };
}

function send(state: ConversationState, id: string, text: string): ConversationState {
  const trimmed = text.trim();
  if (trimmed.length === 0 || state.pending !== "idle") {
    return state;
  }
  return withItems(state, [...state.items, { kind: "user", id, text: trimmed, status: "sending" }], true);
}

/** Reducer da conversa (US1, US3). Puro: mesma entrada, mesma saída; nunca muta `state`. */
export function chatReducer(state: ConversationState, action: ConversationAction): ConversationState {
  switch (action.type) {
    case "send":
      return send(state, action.id, action.text);

    case "received": {
      if (state.pending !== "sending") {
        return state;
      }
      const { result } = action;
      const items = settleSendingUser(state.items, "delivered");
      const item: ConversationItem =
        "status" in result
          ? { kind: "approval", id: action.id, approval: result.approval, run: runOf(result), card: initialApprovalCard }
          : { kind: "assistant", id: action.id, run: runOf(result) };
      return {
        ...withItems(state, [...items, item], false),
        conversationId: state.conversationId ?? result.conversationId,
      };
    }

    case "failed": {
      if (state.pending !== "sending") {
        return state;
      }
      const user = [...state.items].reverse().find((item): item is UserItem => item.kind === "user" && item.status === "sending");
      if (!user) {
        return withItems(state, state.items, false);
      }
      const items = settleSendingUser(state.items, "failed");
      const error: ErrorItem = { kind: "error", id: action.id, error: action.error, retryText: user.text, userItemId: user.id };
      return withItems(state, [...items, error], false);
    }

    case "retry": {
      const target = state.items.find((item): item is ErrorItem => item.kind === "error" && item.id === action.errorItemId);
      if (!target || state.pending !== "idle") {
        return state;
      }
      const items = state.items.filter((item) => item.id !== target.id && item.id !== target.userItemId);
      return send(withItems(state, items, false), action.id, target.retryText);
    }

    case "reset":
      return initialConversation;

    case "approvalUpdated": {
      const items = state.items.map((item) =>
        item.kind === "approval" && item.id === action.itemId ? { ...item, card: approvalReducer(item.card, action.action) } : item,
      );
      return withItems(state, items, state.pending === "sending");
    }

    case "decisionReceived": {
      const index = state.items.findIndex((item) => item.kind === "approval" && item.id === action.itemId);
      if (index === -1) {
        return state;
      }
      const assistant: AssistantItem = { kind: "assistant", id: action.id, run: runOf(action.data) };
      const items = [...state.items.slice(0, index + 1), assistant, ...state.items.slice(index + 1)];
      return withItems(state, items, state.pending === "sending");
    }
  }
}
