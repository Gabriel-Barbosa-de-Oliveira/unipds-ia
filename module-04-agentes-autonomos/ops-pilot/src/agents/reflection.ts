import { z } from "zod";

import { createModel } from "./model.ts";
import { startTimer } from "./metrics.ts";
import type { ReasoningStrategy, RunOptions, RunResult, TraceEvent } from "./types.ts";

const DEFAULT_MAX_REFLECTIONS = 2;

const CRITIC_PROMPT =
  "Você é um crítico rigoroso de respostas de um copiloto de operações. Avalie a resposta final " +
  "APENAS contra as observações do trace fornecidas — não invente critérios novos, não avalie " +
  "estilo. Aprove quando a resposta é consistente com as observações e responde ao pedido; " +
  "reprove quando a resposta contradiz, ignora ou extrapola o que foi observado.";

const verdictSchema = z.object({
  approved: z.boolean(),
  feedback: z
    .string()
    .min(1)
    .describe(
      "Motivo da decisão. Se reprovado: o que corrigir, de forma específica e acionável.",
    ),
});

export type Verdict = z.infer<typeof verdictSchema>;

export interface ReflectionOptions {
  maxReflections?: number;
}

type RunAttempt = (input: string, options?: RunOptions) => Promise<RunResult>;
type CritiqueFn = (input: string, result: RunResult) => Promise<Verdict>;

/** Extrai apenas os resultados das observações de um trace — é contra isso que o crítico avalia a resposta (FR-002). */
export function observationsOf(trace: readonly TraceEvent[]): unknown[] {
  return trace.filter((event) => event.type === "observation").map((event) => event.result);
}

/** Monta as mensagens (system + user) enviadas ao crítico. Função pura, determinística. */
export function buildCritiqueMessages(
  input: string,
  trace: readonly TraceEvent[],
  answer: string,
): [string, string][] {
  return [
    ["system", CRITIC_PROMPT],
    [
      "user",
      `Pedido: ${input}\nObservações: ${JSON.stringify(observationsOf(trace))}\nResposta: ${answer}`,
    ],
  ];
}

/** Monta a entrada de uma tentativa regenerada, incorporando a tentativa anterior e o feedback do crítico. */
export function buildRetryInput(
  originalInput: string,
  previousAnswer: string,
  feedback: string,
): string {
  return [
    originalInput,
    "",
    `Sua tentativa anterior foi: "${previousAnswer}"`,
    `Ela foi reprovada por um crítico pelo seguinte motivo: ${feedback}`,
    "Gere uma nova resposta que corrija esse problema.",
  ].join("\n");
}

async function critique(input: string, result: RunResult): Promise<Verdict> {
  const verdict = await createModel()
    .withStructuredOutput(verdictSchema)
    .invoke(buildCritiqueMessages(input, result.trace, result.answer));

  if (!verdict) {
    throw new Error(
      "Crítico não retornou uma avaliação estruturada válida (o modelo não chamou a ferramenta esperada)",
    );
  }

  return verdict;
}

interface ReflectionResult {
  answer: string;
  trace: TraceEvent[];
  llmCalls: number;
}

/**
 * Orquestração pura do ciclo de reflection: recebe as chamadas de tentativa/crítica já resolvidas
 * (nenhuma IO própria) e decide quantas tentativas rodar. `maxReflections` conta regenerações
 * extras após a 1ª tentativa — o número máximo de tentativas avaliadas é `maxReflections + 1`.
 */
export async function runReflectionLoop(
  runAttempt: RunAttempt,
  critiqueFn: CritiqueFn,
  input: string,
  options: RunOptions | undefined,
  maxReflections: number,
): Promise<ReflectionResult> {
  let trace: TraceEvent[] = [];
  let llmCalls = 0;
  let currentInput = input;
  let regenerationsDone = 0;

  while (true) {
    const attempt = await runAttempt(currentInput, options);
    const offset = trace.length;
    trace = trace.concat(attempt.trace.map((event, index) => ({ ...event, at: offset + index })));
    llmCalls += attempt.metrics.llmCalls;

    const verdict = await critiqueFn(input, attempt);
    llmCalls += 1;
    trace = trace.concat([{ type: "critique", at: trace.length, content: verdict.feedback }]);

    if (verdict.approved || regenerationsDone >= maxReflections) {
      return { answer: attempt.answer, trace, llmCalls };
    }

    regenerationsDone += 1;
    currentInput = buildRetryInput(input, attempt.answer, verdict.feedback);
  }
}

/**
 * Decora qualquer ReasoningStrategy com um ciclo de autocrítica e regeneração, sem alterar a
 * estratégia envolvida. `opts.maxReflections` (padrão 2) limita as regenerações extras.
 */
export function withReflection(
  strategy: ReasoningStrategy,
  opts?: ReflectionOptions,
): ReasoningStrategy {
  const maxReflections = opts?.maxReflections ?? DEFAULT_MAX_REFLECTIONS;

  return {
    name: `reflect:${strategy.name}`,

    async run(input: string, options?: RunOptions): Promise<RunResult> {
      const elapsed = startTimer();
      const result = await runReflectionLoop(
        (attemptInput, attemptOptions) => strategy.run(attemptInput, attemptOptions),
        critique,
        input,
        options,
        maxReflections,
      );

      return {
        answer: result.answer,
        trace: result.trace,
        metrics: { llmCalls: result.llmCalls, latencyMs: elapsed() },
      };
    },
  };
}
