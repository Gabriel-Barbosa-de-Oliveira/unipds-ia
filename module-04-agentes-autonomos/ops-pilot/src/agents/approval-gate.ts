import { tool, type StructuredToolInterface } from "@langchain/core/tools";
import { z } from "zod";

import { isGatedTool, type GatedToolName } from "../domain/approval.ts";
import type { OpsStoreRepository } from "../services/ops-store.repository.ts";
import { createOpsTools, openIncidentShape, resolveIncidentShape, toStructuredError } from "./tools.ts";

/**
 * Porta de aprovação por requisição (spec 015, research.md item 1). As ferramentas que mudam a
 * produção não executam: registram a proposta aqui e o /chat responde 202. Exceção lançada pela
 * ferramenta seria engolida pelo `ToolNode` (`handleToolErrors`), por isso a porta é um registro.
 */
export interface ApprovalGate {
  proposed?: { tool: GatedToolName; args: Record<string, unknown> };
}

export function createApprovalGate(): ApprovalGate {
  return {};
}

const GATED_SCHEMAS = {
  open_incident: z.object(openIncidentShape),
  resolve_incident: z.object(resolveIncidentShape),
} satisfies Record<GatedToolName, z.ZodTypeAny>;

/** Observação devolvida ao modelo quando a ação fica aguardando aprovação. */
export const AWAITING_APPROVAL_OBSERVATION = {
  status: "awaiting_approval",
  message:
    "Ação registrada e aguardando aprovação humana; ela NÃO foi executada. Não tente de novo nem afirme que " +
    "foi feita: informe ao plantonista que a ação aguarda aprovação.",
} as const;

export const REJECTED_OBSERVATION = {
  status: "rejected",
  message: "Já existe uma ação aguardando aprovação nesta requisição; só uma por vez. Nada foi executado.",
} as const;

/**
 * Instâncias com porta criadas por `createGatedOpsTools`. A equipe (017) confere por identidade que o
 * executor só recebe estas — nenhuma composição entrega a ele uma ferramenta que executa direto.
 */
const GATED_INSTANCES = new WeakSet<StructuredToolInterface>();

export function isApprovalGated(candidate: StructuredToolInterface): boolean {
  return GATED_INSTANCES.has(candidate);
}

/** Revalida os args guardados antes de executar (Princípio II) — mesmos schemas das ferramentas. */
export function parseGatedArgs(toolName: GatedToolName, args: unknown): Record<string, unknown> {
  return GATED_SCHEMAS[toolName].parse(args) as Record<string, unknown>;
}

/**
 * Mesmas 5 ferramentas de `createOpsTools`, com `open_incident`/`resolve_incident` trocadas por
 * versões que só registram a proposta no `gate` (mesmo nome, descrição e schema).
 */
export function createGatedOpsTools(store: OpsStoreRepository, gate: ApprovalGate): StructuredToolInterface[] {
  return createOpsTools(store).map((original) => {
    const name = original.name;
    if (!isGatedTool(name)) {
      return original;
    }
    const gated = tool(
      async (args: Record<string, unknown>) => {
        if (gate.proposed) {
          return JSON.stringify(REJECTED_OBSERVATION);
        }
        gate.proposed = { tool: name, args };
        return JSON.stringify(AWAITING_APPROVAL_OBSERVATION);
      },
      { name, description: original.description, schema: GATED_SCHEMAS[name] },
    );
    GATED_INSTANCES.add(gated);
    return gated;
  });
}

/** Executa a ação aprovada direto no store; erro de domínio vira resultado estruturado. */
export async function executeGatedAction(
  store: OpsStoreRepository,
  toolName: GatedToolName,
  args: Record<string, unknown>,
): Promise<unknown> {
  try {
    if (toolName === "resolve_incident") {
      const { id, summary } = GATED_SCHEMAS.resolve_incident.parse(args);
      return await store.resolveIncident(id, summary);
    }
    const input = GATED_SCHEMAS.open_incident.parse(args);
    return await store.openIncident(input);
  } catch (error) {
    return toStructuredError(error, toolName === "resolve_incident" ? { id: args.id } : { service: args.service });
  }
}
