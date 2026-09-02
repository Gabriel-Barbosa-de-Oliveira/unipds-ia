import { GraphRecursionError } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import type { BaseMessage } from "@langchain/core/messages";

import { lastAnswer, messagesToTrace } from "./message-trace.ts";
import { createModel } from "./model.ts";
import { buildMetrics, LlmCallCounter, startTimer } from "./metrics.ts";
import { opsTools } from "./tools.ts";
import type { ReasoningStrategy, RunOptions, RunResult } from "./types.ts";

const DEFAULT_MAX_ITERATIONS = 8;
const LIMIT_REACHED_ANSWER =
  "Não foi possível concluir dentro do limite de passos configurado; encerrando de forma controlada.";

export const reactStrategy: ReasoningStrategy = {
  name: "react",

  async run(input: string, options?: RunOptions): Promise<RunResult> {
    const elapsed = startTimer();
    const counter = new LlmCallCounter();
    const maxIterations = options?.maxIterations ?? DEFAULT_MAX_ITERATIONS;

    const agent = createReactAgent({
      llm: createModel(),
      tools: opsTools,
    });

    let lastMessages: BaseMessage[] = [];

    try {
      const stream = await agent.stream(
        { messages: [{ role: "user", content: input }] },
        { recursionLimit: maxIterations, callbacks: [counter], streamMode: "values" },
      );

      for await (const chunk of stream) {
        lastMessages = chunk.messages;
      }

      const trace = messagesToTrace(lastMessages);
      return {
        answer: lastAnswer(trace) ?? LIMIT_REACHED_ANSWER,
        trace,
        metrics: buildMetrics(counter, elapsed()),
      };
    } catch (error) {
      if (error instanceof GraphRecursionError) {
        // Guardrail: limite de passos atingido sem resposta final — encerra de forma
        // controlada com o trace parcial acumulado até aqui, conforme o contrato de
        // ReasoningStrategy (FR-006).
        const trace = messagesToTrace(lastMessages);
        trace.push({ type: "answer", at: trace.length, content: LIMIT_REACHED_ANSWER });
        return { answer: LIMIT_REACHED_ANSWER, trace, metrics: buildMetrics(counter, elapsed()) };
      }
      throw error;
    }
  },
};
