import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import type { BaseMessage } from "@langchain/core/messages";
import { z } from "zod";

import { lastAnswer, messagesToTrace } from "./message-trace.ts";
import { createModel } from "./model.ts";
import { buildMetrics, LlmCallCounter, startTimer } from "./metrics.ts";
import { opsTools } from "./tools.ts";
import type { ReasoningStrategy, RunOptions, RunResult, TraceEvent } from "./types.ts";

/** Corte físico de passos executados — nunca ultrapassado, mesmo se o replanner insistir (FR-005). */
const HARD_STEP_CAP = 8;
const LIMIT_REACHED_ANSWER = `Não foi possível concluir dentro do limite de ${HARD_STEP_CAP} passos; encerrando de forma controlada.`;

const PlanSchema = z.object({
  steps: z
    .array(z.string().min(1))
    .min(1)
    .describe("Lista ordenada de passos necessários para atender ao pedido do plantonista"),
});

const ReplanSchema = z.union([
  z.object({
    action: z.literal("response"),
    response: z.string().min(1).describe("Resposta final para o plantonista"),
  }),
  z.object({
    action: z.literal("plan"),
    steps: z.array(z.string().min(1)).min(1).describe("Passos restantes revisados"),
  }),
]);

const PlanExecuteState = Annotation.Root({
  input: Annotation<string>,
  plan: Annotation<string[]>({ reducer: (_prev, next) => next, default: () => [] }),
  pastSteps: Annotation<{ step: string; result: string }[]>({
    reducer: (prev, next) => prev.concat(next),
    default: () => [],
  }),
  response: Annotation<string | undefined>({ reducer: (_prev, next) => next, default: () => undefined }),
  trace: Annotation<TraceEvent[]>({ reducer: (prev, next) => prev.concat(next), default: () => [] }),
  stepsTaken: Annotation<number>({ reducer: (_prev, next) => next, default: () => 0 }),
});

type PlanExecuteStateType = typeof PlanExecuteState.State;

function buildReplanPrompt(state: PlanExecuteStateType): string {
  const done = state.pastSteps
    .map((entry, index) => `${index + 1}. ${entry.step} -> ${entry.result}`)
    .join("\n");
  const remaining = state.plan.length > 0 ? state.plan.map((step) => `- ${step}`).join("\n") : "(nenhum)";

  return [
    `Pedido original do plantonista: ${state.input}`,
    "",
    "Passos já executados e seus resultados:",
    done.length > 0 ? done : "(nenhum ainda)",
    "",
    "Passos restantes no plano atual:",
    remaining,
    "",
    "Se os passos executados já respondem completamente ao pedido, responda com action=\"response\" " +
      "e a resposta final para o plantonista. Caso contrário, responda com action=\"plan\" e a lista " +
      "revisada dos passos que ainda faltam.",
  ].join("\n");
}

/** Constrói o grafo Plan-and-Execute (planner → executor → replanner), com corte em `stepCap` passos. */
function buildGraph(counter: LlmCallCounter, stepCap: number, noReplanner: boolean) {
  const model = createModel();
  const plannerModel = model.withStructuredOutput(PlanSchema);
  const replannerModel = model.withStructuredOutput(ReplanSchema);
  const executorAgent = createReactAgent({ llm: model, tools: opsTools });

  async function planner(state: PlanExecuteStateType): Promise<Partial<PlanExecuteStateType>> {
    const result = await plannerModel.invoke([{ role: "user", content: state.input }], {
      callbacks: [counter],
    });

    if (!result) {
      throw new Error(
        "Planner não retornou um plano estruturado válido (o modelo configurado em OPENROUTER_MODEL " +
          "não chamou a tool de saída estruturada esperada)",
      );
    }

    return {
      plan: result.steps,
      trace: [{ type: "plan", at: Date.now(), steps: result.steps }],
    };
  }

  async function executor(state: PlanExecuteStateType): Promise<Partial<PlanExecuteStateType>> {
    const [step, ...rest] = state.plan;
    if (!step) {
      return {};
    }

    const stream = await executorAgent.stream(
      { messages: [{ role: "user", content: step }] },
      { callbacks: [counter], streamMode: "values" },
    );

    let messages: BaseMessage[] = [];
    for await (const chunk of stream) {
      messages = chunk.messages;
    }

    const stepTrace = messagesToTrace(messages, state.trace.length);
    const resultText = lastAnswer(stepTrace) ?? "(sem resposta da tool)";

    return {
      plan: rest,
      pastSteps: [{ step, result: resultText }],
      trace: stepTrace,
      stepsTaken: state.stepsTaken + 1,
    };
  }

  async function replanner(state: PlanExecuteStateType): Promise<Partial<PlanExecuteStateType>> {
    const result = await replannerModel.invoke(
      [{ role: "user", content: buildReplanPrompt(state) }],
      { callbacks: [counter] },
    );

    if (!result) {
      throw new Error(
        "Replanner não retornou uma decisão estruturada válida (o modelo configurado em OPENROUTER_MODEL " +
          "não chamou a tool de saída estruturada esperada)",
      );
    }

    if (result.action === "response") {
      return {
        response: result.response,
        trace: [{ type: "answer", at: state.trace.length, content: result.response }],
      };
    }

    return {
      plan: result.steps,
      trace: [
        {
          type: "critique",
          at: state.trace.length,
          content: `Plano revisado: ${result.steps.join(" | ")}`,
        },
      ],
    };
  }

  async function giveUp(state: PlanExecuteStateType): Promise<Partial<PlanExecuteStateType>> {
    return {
      response: LIMIT_REACHED_ANSWER,
      trace: [{ type: "answer", at: state.trace.length, content: LIMIT_REACHED_ANSWER }],
    };
  }

  /** Modo --no-replanner: sintetiza a resposta a partir dos passos executados, sem chamada de LLM. */
  async function finalize(state: PlanExecuteStateType): Promise<Partial<PlanExecuteStateType>> {
    const response =
      state.pastSteps.length > 0
        ? state.pastSteps.map((entry) => `${entry.step}: ${entry.result}`).join("\n")
        : "Nenhum passo foi executado.";

    return {
      response,
      trace: [{ type: "answer", at: state.trace.length, content: response }],
    };
  }

  function afterReplan(state: PlanExecuteStateType): "executor" | "giveUp" | typeof END {
    if (state.response !== undefined) {
      return END;
    }
    if (state.stepsTaken >= stepCap) {
      return "giveUp";
    }
    return "executor";
  }

  /** Sem replanner: segue o plano inicial até o fim ou até o corte de passos, sem reavaliar via LLM. */
  function afterExecutor(state: PlanExecuteStateType): "executor" | "replanner" | "finalize" | "giveUp" {
    if (!noReplanner) {
      return "replanner";
    }
    if (state.stepsTaken >= stepCap) {
      return "giveUp";
    }
    return state.plan.length > 0 ? "executor" : "finalize";
  }

  return new StateGraph(PlanExecuteState)
    .addNode("planner", planner)
    .addNode("executor", executor)
    .addNode("replanner", replanner)
    .addNode("finalize", finalize)
    .addNode("giveUp", giveUp)
    .addEdge(START, "planner")
    .addEdge("planner", "executor")
    .addConditionalEdges("executor", afterExecutor, ["executor", "replanner", "finalize", "giveUp"])
    .addConditionalEdges("replanner", afterReplan, ["executor", "giveUp", END])
    .addEdge("finalize", END)
    .addEdge("giveUp", END)
    .compile();
}

export const planAndExecuteStrategy: ReasoningStrategy = {
  name: "plan-and-execute",

  async run(input: string, options?: RunOptions): Promise<RunResult> {
    const elapsed = startTimer();
    const counter = new LlmCallCounter();
    const stepCap = Math.min(options?.maxIterations ?? HARD_STEP_CAP, HARD_STEP_CAP);
    const noReplanner = options?.noReplanner ?? false;

    const graph = buildGraph(counter, stepCap, noReplanner);
    const result = await graph.invoke(
      { input, plan: [], pastSteps: [], trace: [], stepsTaken: 0, response: undefined },
      { recursionLimit: 2 + stepCap * 2 },
    );

    return {
      answer: result.response ?? LIMIT_REACHED_ANSWER,
      trace: result.trace,
      metrics: buildMetrics(counter, elapsed()),
    };
  },
};
