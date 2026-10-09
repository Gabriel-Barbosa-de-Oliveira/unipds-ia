import type { ApprovalDecision, PendingAction } from "../domain/approval.ts";

export type DecideResult =
  | { ok: true; action: PendingAction }
  | { ok: false; reason: "not_found" }
  | { ok: false; reason: "already_decided" | "expired"; action: PendingAction };

/**
 * Contrato de persistência das ações aguardando aprovação (spec 015), independente do adaptador
 * — mesmo padrão de `RequestStore`.
 */
export interface ApprovalStore {
  create(action: PendingAction): Promise<void>;

  find(id: string): Promise<PendingAction | undefined>;

  /** Grava a decisão só se a ação ainda estiver pendente e no prazo — no máximo uma decisão (FR-019). */
  decide(id: string, decision: ApprovalDecision, now: Date, decisionRequestId: string): Promise<DecideResult>;
}
