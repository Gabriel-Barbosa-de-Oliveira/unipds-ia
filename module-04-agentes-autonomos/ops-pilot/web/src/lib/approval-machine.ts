import { unavailableReason, type UiError } from "./errors.ts";

export type Decision = "approve" | "deny";

/** Estado do cartão Aprovar/Negar (data-model.md, ApprovalCard). */
export type ApprovalCardState =
  | { status: "pending"; error?: UiError }
  | { status: "submitting"; decision: Decision }
  | { status: "approved" }
  | { status: "denied" }
  | { status: "unavailable"; reason: "Já decidida" | "Expirou" | "Não encontrada" };

export type ApprovalCardAction =
  | { type: "decide"; decision: Decision }
  | { type: "succeeded"; status: "approved" | "denied" }
  | { type: "failed"; error: UiError };

export const initialApprovalCard: ApprovalCardState = { status: "pending" };

/**
 * Transições do cartão. Pura. Só `pending` aceita decisão (sem decidir em dobro, FR-016/017);
 * 409/410/404 tornam o cartão indisponível, qualquer outra falha volta a `pending` com o erro.
 */
export function approvalReducer(state: ApprovalCardState, action: ApprovalCardAction): ApprovalCardState {
  switch (action.type) {
    case "decide":
      return state.status === "pending" ? { status: "submitting", decision: action.decision } : state;
    case "succeeded":
      return state.status === "submitting" ? { status: action.status } : state;
    case "failed": {
      if (state.status !== "submitting") {
        return state;
      }
      const reason = unavailableReason(action.error);
      return reason ? { status: "unavailable", reason } : { status: "pending", error: action.error };
    }
  }
}

/** O cartão ainda espera uma decisão humana — bloqueia o envio de novas mensagens (FR-018). */
export function isAwaitingDecision(state: ApprovalCardState): boolean {
  return state.status === "pending" || state.status === "submitting";
}
