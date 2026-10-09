import { GraphRecursionError } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import type { BaseMessage } from "@langchain/core/messages";
import type { StructuredToolInterface } from "@langchain/core/tools";

import { UsageCollector } from "../context/tokens.ts";
import { lastAnswer, messagesToTrace } from "./message-trace.ts";
import { buildMetrics, LlmCallCounter, startTimer } from "./metrics.ts";
import { loadModelConfig, toolCallingModel } from "./model.ts";
import { ModelUsageTracker, summarizeModelUsage, withModelFallbacks } from "./model-usage.ts";
import { opsTools } from "./tools.ts";
import type { ReasoningStrategy, RunOptions, RunResult } from "./types.ts";

const DEFAULT_MAX_ITERATIONS = 8;
const LIMIT_REACHED_ANSWER =
  "Não foi possível concluir dentro do limite de passos configurado; encerrando de forma controlada.";

/**
 * Fábrica da estratégia ReAct, fechada sobre `tools` — permite compor a estratégia sobre um
 * conjunto de tools diferente do padrão (`opsTools`, sobre `SqliteOpsStore`), como faz
 * `src/bench.ts` sobre um mock em memória isolado e reprodutível (research.md item 2).
 */
export function createReactStrategy(tools: StructuredToolInterface[]): ReasoningStrategy {
  return {
    name: "react",

    async run(input: string, options?: RunOptions): Promise<RunResult> {
      const elapsed = startTimer();
      const counter = new LlmCallCounter();
      const usageCollector = new UsageCollector();
      const modelTracker = new ModelUsageTracker();
      const modelConfig = loadModelConfig(process.env);
      const maxIterations = options?.maxIterations ?? DEFAULT_MAX_ITERATIONS;

      const agent = createReactAgent({
        llm: toolCallingModel(tools, modelConfig),
        tools,
      });

      let lastMessages: BaseMessage[] = [];

      try {
        const stream = await agent.stream(
          { messages: [{ role: "user", content: input }] },
          { recursionLimit: maxIterations, callbacks: [counter, usageCollector, modelTracker], streamMode: "values" },
        );

        for await (const chunk of stream) {
          lastMessages = chunk.messages;
        }

        const usage = summarizeModelUsage(modelTracker.log, modelConfig);
        const trace = withModelFallbacks(messagesToTrace(lastMessages), usage.fallbacks);
        return {
          answer: lastAnswer(trace) ?? LIMIT_REACHED_ANSWER,
          trace,
          metrics: buildMetrics(counter, usageCollector, elapsed(), usage.modelUsed),
        };
      } catch (error) {
        if (error instanceof GraphRecursionError) {
          // Guardrail: limite de passos atingido sem resposta final — encerra de forma
          // controlada com o trace parcial acumulado até aqui, conforme o contrato de
          // ReasoningStrategy (FR-006).
          const usage = summarizeModelUsage(modelTracker.log, modelConfig);
          const partial = messagesToTrace(lastMessages);
          partial.push({ type: "answer", at: partial.length, content: LIMIT_REACHED_ANSWER });
          const trace = withModelFallbacks(partial, usage.fallbacks);
          return {
            answer: LIMIT_REACHED_ANSWER,
            trace,
            metrics: buildMetrics(counter, usageCollector, elapsed(), usage.modelUsed),
          };
        }
        throw error;
      }
    },
  };
}

/** Composição padrão, usada por `agents/index.ts`/`http/server.ts`/`arena.ts` — sobre `opsTools`. */
export const reactStrategy: ReasoningStrategy = createReactStrategy(opsTools);
