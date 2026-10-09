import type { StructuredToolInterface } from "@langchain/core/tools";

import { isApprovalGated } from "../agents/approval-gate.ts";
import { buildMetrics, LlmCallCounter, startTimer } from "../agents/metrics.ts";
import { loadModelConfig, type ModelConfig } from "../agents/model.ts";
import { ModelUsageTracker, summarizeModelUsage, withModelFallbacks } from "../agents/model-usage.ts";
import type { ReasoningStrategy, RunResult } from "../agents/types.ts";
import { UsageCollector } from "../context/tokens.ts";
import { TeamToolsNotGatedError } from "../domain/errors.ts";
import { createModelTeamDeps } from "./members.ts";
import { selectRoleTools } from "./roles.ts";
import { runTeam, type TeamDeps } from "./team-graph.ts";

/** Modo equipe (spec 017): supervisor + analista, planejador e executor sobre um quadro compartilhado. */
export type { HandoffTarget, TeamRole } from "../agents/types.ts";

function isComplete(deps: Partial<TeamDeps>): deps is TeamDeps {
  return Boolean(deps.decide && deps.runAnalyst && deps.runPlanner && deps.runExecutor);
}

/**
 * Garante que o executor só recebe ferramentas com porta de aprovação (FR-010). Falha na montagem —
 * nunca existe uma equipe capaz de executar sem decisão humana.
 */
export function assertExecutorGated(tools: readonly StructuredToolInterface[]): void {
  const ungated = selectRoleTools(tools, "executor").filter((tool) => !isApprovalGated(tool));
  if (ungated.length > 0) {
    throw new TeamToolsNotGatedError(ungated.map((tool) => tool.name));
  }
}

/**
 * A equipe como `ReasoningStrategy` (rota `team`). `deps` substitui supervisor/papéis — testes
 * injetam fakes; ausentes, usam os modelos reais (`members.ts`), criados só na execução.
 */
export function createTeamStrategy(tools: readonly StructuredToolInterface[], deps: Partial<TeamDeps> = {}): ReasoningStrategy {
  assertExecutorGated(tools);
  const injected = isComplete(deps) ? deps : undefined;

  return {
    name: "team",

    async run(input: string): Promise<RunResult> {
      const elapsed = startTimer();
      const counter = new LlmCallCounter();
      const usageCollector = new UsageCollector();
      const modelTracker = new ModelUsageTracker();
      // Com todos os papéis injetados não há chamada a modelo — e o env do modelo não é exigido.
      const modelConfig: ModelConfig = injected ? { primary: "team-injected" } : loadModelConfig(process.env);
      const teamDeps = injected ?? { ...createModelTeamDeps(tools, modelConfig), ...deps };

      const result = await runTeam(input, teamDeps, [counter, usageCollector, modelTracker]);
      const usage = summarizeModelUsage(modelTracker.log, modelConfig);
      return {
        answer: result.answer,
        trace: withModelFallbacks(result.trace, usage.fallbacks),
        metrics: buildMetrics(counter, usageCollector, elapsed(), usage.modelUsed),
      };
    },
  };
}
