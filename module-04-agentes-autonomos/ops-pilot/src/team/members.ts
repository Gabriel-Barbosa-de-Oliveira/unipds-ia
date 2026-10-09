import type { Callbacks } from "@langchain/core/callbacks/manager";
import type { BaseMessage } from "@langchain/core/messages";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import type { z } from "zod";

import { lastAnswer, messagesToTrace } from "../agents/message-trace.ts";
import { createModel, toolCallingModel, type ModelConfig } from "../agents/model.ts";
import { formatTrace } from "../agents/trace.ts";
import type { TraceEvent } from "../agents/types.ts";
import { renderBlackboard } from "./blackboard.ts";
import {
  ANALYST_EXTRACTION_PROMPT,
  ANALYST_PROMPT,
  AnalystReportSchema,
  EXECUTOR_PROMPT,
  PLANNER_PROMPT,
  PlanSchema,
  selectRoleTools,
} from "./roles.ts";
import { nextSchema } from "./supervisor.ts";
import type { RoleInput, TeamDeps } from "./team-graph.ts";

/** Passos de um papel com ferramentas por turno (agente ReAct interno). */
const ROLE_RECURSION_LIMIT = 8;

/**
 * Implementações reais do supervisor e dos papéis (spec 017) — o único módulo da equipe com IO de
 * modelo. Cada papel recebe só as ferramentas da sua lista (`selectRoleTools`).
 */

function roleMessages(systemPrompt: string, input: RoleInput): [string, string][] {
  return [
    ["system", systemPrompt],
    ["user", `Instrução do supervisor: ${input.brief}\n\n${renderBlackboard(input.blackboard)}`],
  ];
}

async function runToolAgent(
  tools: StructuredToolInterface[],
  config: ModelConfig,
  messages: [string, string][],
  callbacks?: Callbacks,
): Promise<TraceEvent[]> {
  const agent = createReactAgent({ llm: toolCallingModel(tools, config), tools });
  const stream = await agent.stream(
    { messages: messages.map(([role, content]) => ({ role, content })) },
    { callbacks, recursionLimit: ROLE_RECURSION_LIMIT, streamMode: "values" },
  );
  let last: BaseMessage[] = [];
  for await (const chunk of stream) {
    last = chunk.messages;
  }
  return messagesToTrace(last);
}

export function createModelTeamDeps(tools: readonly StructuredToolInterface[], config: ModelConfig): TeamDeps {
  const analystTools = selectRoleTools(tools, "analista");
  const executorTools = selectRoleTools(tools, "executor");
  const supervisorModel = createModel((model) => model.withStructuredOutput<z.infer<typeof nextSchema>>(nextSchema), config);
  const extractor = createModel(
    (model) => model.withStructuredOutput<z.infer<typeof AnalystReportSchema>>(AnalystReportSchema),
    config,
  );
  // Planejador: só saída estruturada — nenhum `bindTools` (FR-008).
  const plannerModel = createModel((model) => model.withStructuredOutput<z.infer<typeof PlanSchema>>(PlanSchema), config);

  return {
    async decide(messages, callbacks) {
      return supervisorModel.invoke(messages, { callbacks });
    },

    async runAnalyst(input) {
      const trace = await runToolAgent(analystTools, config, roleMessages(ANALYST_PROMPT, input), input.callbacks);
      const report = await extractor.invoke(
        [
          ["system", ANALYST_EXTRACTION_PROMPT],
          ["user", formatTrace(trace) || "(sem observações)"],
        ],
        { callbacks: input.callbacks },
      );
      // Só os fatos validados chegam ao quadro; o texto livre do analista fica só no trace.
      const parsed = AnalystReportSchema.safeParse(report);
      return { trace, facts: parsed.success ? parsed.data.facts : [] };
    },

    async runPlanner(input) {
      const raw = await plannerModel.invoke(roleMessages(PLANNER_PROMPT, input), { callbacks: input.callbacks });
      const parsed = PlanSchema.safeParse(raw);
      const steps = parsed.success ? parsed.data.steps : [];
      return { trace: steps.length > 0 ? [{ type: "plan", at: 0, steps }] : [], steps };
    },

    async runExecutor(input) {
      const trace = await runToolAgent(executorTools, config, roleMessages(EXECUTOR_PROMPT, input), input.callbacks);
      return { trace, summary: lastAnswer(trace) ?? "" };
    },
  };
}
