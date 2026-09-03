import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildCritiqueMessages,
  buildRetryInput,
  observationsOf,
  runReflectionLoop,
} from "./reflection.ts";
import type { RunResult, TraceEvent } from "./types.ts";

function fakeAttempt(answer: string, llmCalls = 1): RunResult {
  const trace: TraceEvent[] = [
    { type: "action", at: 0, tool: "list_alerts", args: { status: "firing" } },
    { type: "observation", at: 1, result: [{ id: "alert-1", status: "firing" }] },
    { type: "answer", at: 2, content: answer },
  ];
  return { answer, trace, metrics: { llmCalls, latencyMs: 10 } };
}

test("observationsOf extrai apenas os resultados dos eventos observation", () => {
  const trace: TraceEvent[] = [
    { type: "thought", at: 0, content: "pensando" },
    { type: "observation", at: 1, result: { count: 3 } },
    { type: "answer", at: 2, content: "resposta" },
  ];
  assert.deepEqual(observationsOf(trace), [{ count: 3 }]);
});

test("buildCritiqueMessages monta system + user com pedido, observações e resposta", () => {
  const trace: TraceEvent[] = [{ type: "observation", at: 0, result: { firing: 3 } }];
  const messages = buildCritiqueMessages("quais alertas estão firing?", trace, "há 3 alertas");

  assert.equal(messages.length, 2);
  assert.equal(messages[0]![0], "system");
  assert.equal(messages[1]![0], "user");
  assert.match(messages[1]![1], /Pedido: quais alertas estão firing\?/);
  assert.match(messages[1]![1], /Observações: \[\{"firing":3\}\]/);
  assert.match(messages[1]![1], /Resposta: há 3 alertas/);
});

test("buildRetryInput incorpora a tentativa anterior e o feedback do crítico", () => {
  const retryInput = buildRetryInput("pedido original", "resposta errada", "faltou considerar X");

  assert.match(retryInput, /^pedido original/);
  assert.match(retryInput, /resposta errada/);
  assert.match(retryInput, /faltou considerar X/);
});

test("runReflectionLoop: aprovação já na 1ª tentativa não gera regeneração", async () => {
  let attemptCalls = 0;
  const runAttempt = async () => {
    attemptCalls += 1;
    return fakeAttempt("resposta correta");
  };
  const critique = async () => ({ approved: true, feedback: "consistente com as observações" });

  const result = await runReflectionLoop(runAttempt, critique, "pedido", undefined, 2);

  assert.equal(attemptCalls, 1);
  assert.equal(result.answer, "resposta correta");
  assert.equal(result.trace.filter((e) => e.type === "critique").length, 1);
  assert.equal(result.llmCalls, 1 /* tentativa */ + 1 /* crítica */);
});

test("runReflectionLoop: reprovação seguida de aprovação regenera com o feedback no contexto", async () => {
  const receivedInputs: string[] = [];
  let attemptCalls = 0;
  const runAttempt = async (input: string) => {
    receivedInputs.push(input);
    attemptCalls += 1;
    return fakeAttempt(attemptCalls === 1 ? "resposta incompleta" : "resposta corrigida");
  };
  const critique = async (_input: string, result: RunResult) => {
    if (result.answer === "resposta incompleta") {
      return { approved: false, feedback: "faltou o alerta X" };
    }
    return { approved: true, feedback: "agora está completo" };
  };

  const result = await runReflectionLoop(runAttempt, critique, "pedido original", undefined, 2);

  assert.equal(attemptCalls, 2);
  assert.equal(result.answer, "resposta corrigida");
  assert.match(receivedInputs[1]!, /pedido original/);
  assert.match(receivedInputs[1]!, /resposta incompleta/);
  assert.match(receivedInputs[1]!, /faltou o alerta X/);

  const critiqueEvents = result.trace.filter((e) => e.type === "critique");
  assert.equal(critiqueEvents.length, 2);
  const atValues = result.trace.map((e) => e.at);
  assert.deepEqual(atValues, [...atValues].sort((a, b) => a - b));
  assert.equal(result.llmCalls, 2 /* tentativas */ + 2 /* críticas */);
});

test("runReflectionLoop: esgota maxReflections (padrão) sem aprovação e retorna a última tentativa", async () => {
  let attemptCalls = 0;
  const runAttempt = async () => {
    attemptCalls += 1;
    return fakeAttempt(`tentativa ${attemptCalls}`);
  };
  const critique = async () => ({ approved: false, feedback: "ainda não está certo" });

  const result = await runReflectionLoop(runAttempt, critique, "pedido", undefined, 2);

  assert.equal(attemptCalls, 3 /* maxReflections + 1 */);
  assert.equal(result.answer, "tentativa 3");
  assert.equal(result.trace.filter((e) => e.type === "critique").length, 3);
});

test("runReflectionLoop: maxReflections = 0 nunca regenera, mesmo com reprovação", async () => {
  let attemptCalls = 0;
  const runAttempt = async () => {
    attemptCalls += 1;
    return fakeAttempt("única tentativa");
  };
  const critique = async () => ({ approved: false, feedback: "reprovado" });

  const result = await runReflectionLoop(runAttempt, critique, "pedido", undefined, 0);

  assert.equal(attemptCalls, 1);
  assert.equal(result.answer, "única tentativa");
  assert.equal(result.trace.filter((e) => e.type === "critique").length, 1);
});

test("runReflectionLoop: erro na tentativa propaga em vez de ser mascarado", async () => {
  const runAttempt = async () => {
    throw new Error("falha de infraestrutura");
  };
  const critique = async () => ({ approved: true, feedback: "n/a" });

  await assert.rejects(
    runReflectionLoop(runAttempt, critique, "pedido", undefined, 2),
    /falha de infraestrutura/,
  );
});

test("runReflectionLoop: erro na crítica propaga em vez de ser mascarado", async () => {
  const runAttempt = async () => fakeAttempt("resposta");
  const critique = async () => {
    throw new Error("crítico indisponível");
  };

  await assert.rejects(
    runReflectionLoop(runAttempt, critique, "pedido", undefined, 2),
    /crítico indisponível/,
  );
});
