import assert from "node:assert/strict";
import { describe, test } from "node:test";

import type { TraceEvent } from "../agents/types.ts";
import type { Fact } from "./blackboard.ts";
import { hasProposal, PROPOSAL_ANSWER, runTeam, tagRole, type RoleInput, type TeamDeps } from "./team-graph.ts";

/** Equipe fake e roteirizada: o supervisor devolve as decisões na ordem; os papéis registram o que receberam. */
function fakeTeam(decisions: unknown[], overrides: Partial<TeamDeps> = {}) {
  const calls = { decide: 0, analista: [] as RoleInput[], planejador: [] as RoleInput[], executor: [] as RoleInput[] };
  const facts: Fact[] = [{ statement: "alert-1 firing (critical)", source: "list_alerts" }];

  const deps: TeamDeps = {
    async decide() {
      const decision = decisions[calls.decide] ?? { next: "done", brief: "fim do roteiro" };
      calls.decide += 1;
      if (decision instanceof Error) {
        throw decision;
      }
      return decision;
    },
    async runAnalyst(input) {
      calls.analista.push(input);
      return {
        trace: [
          { type: "action", at: 0, tool: "list_alerts", args: { status: "firing" } },
          { type: "observation", at: 1, result: [{ id: "alert-1" }] },
        ],
        facts,
      };
    },
    async runPlanner(input) {
      calls.planejador.push(input);
      return { trace: [{ type: "plan", at: 0, steps: ["abrir incidente high"] }], steps: ["abrir incidente high"] };
    },
    async runExecutor(input) {
      calls.executor.push(input);
      return { trace: [{ type: "answer", at: 0, content: "feito" }], summary: "feito" };
    },
    ...overrides,
  };
  return { deps, calls };
}

const shape = (trace: readonly TraceEvent[]) =>
  trace.map((event) => (event.type === "handoff" ? `handoff→${event.to}` : `${event.type}@${event.role}`));

describe("runTeam", () => {
  test("analista → planejador → done: passagens na ordem, papéis carimbados e resposta = brief final", async () => {
    const { deps, calls } = fakeTeam([
      { next: "analista", brief: "levante os alertas do checkout" },
      { next: "planejador", brief: "planeje" },
      { next: "done", brief: "Há 1 alerta crítico; plano: abrir incidente." },
    ]);

    const result = await runTeam("investigue o checkout", deps);

    assert.deepEqual(shape(result.trace), [
      "handoff→analista",
      "action@analista",
      "observation@analista",
      "handoff→planejador",
      "plan@planejador",
      "handoff→done",
      "answer@supervisor",
    ]);
    assert.deepEqual(
      result.trace.map((event) => event.at),
      result.trace.map((_, index) => index),
    );
    assert.equal(result.answer, "Há 1 alerta crítico; plano: abrir incidente.");
    assert.equal(result.proposed, false);
    assert.equal(calls.analista[0]?.brief, "levante os alertas do checkout");
    assert.equal(calls.planejador[0]?.blackboard.facts.length, 1, "o planejador vê os fatos do analista");
    assert.deepEqual(result.blackboard.plan, ["abrir incidente high"]);
    assert.equal(result.blackboard.handoffs.length, 3);
  });

  test("pedido só de consulta: analista → done, sem planejador nem executor", async () => {
    const { deps, calls } = fakeTeam([
      { next: "analista", brief: "liste" },
      { next: "done", brief: "1 alerta firing." },
    ]);
    const result = await runTeam("quais alertas?", deps);
    assert.equal(calls.planejador.length, 0);
    assert.equal(calls.executor.length, 0);
    assert.equal(result.answer, "1 alerta firing.");
  });

  test("supervisor em loop: para no teto com resposta de fallback", async () => {
    const { deps, calls } = fakeTeam(Array.from({ length: 20 }, () => ({ next: "analista", brief: "de novo" })));
    const result = await runTeam("x", deps);
    assert.equal(calls.analista.length, 6);
    const last = result.trace.at(-2);
    assert.equal(last?.type === "handoff" && last.to, "done");
    assert.match(last?.type === "handoff" ? last.brief : "", /teto de 6 passagens/);
    assert.match(result.answer, /alert-1 firing/);
  });

  test("decisão inválida no 2º passo aborta com fallback", async () => {
    const { deps } = fakeTeam([{ next: "analista", brief: "liste" }, { next: "redator", brief: "?" }]);
    const result = await runTeam("x", deps);
    assert.match(result.answer, /decisão inválida/);
    assert.match(result.answer, /alert-1 firing/);
  });

  test("supervisor que lança aborta sem propagar a exceção", async () => {
    const { deps } = fakeTeam([new Error("modelo fora")]);
    const result = await runTeam("x", deps);
    assert.match(result.answer, /supervisor falhou \(modelo fora\)/);
    assert.deepEqual(shape(result.trace), ["handoff→done", "answer@supervisor"]);
  });

  test("o mesmo papel chamado duas vezes gera duas passagens", async () => {
    const { deps, calls } = fakeTeam([
      { next: "analista", brief: "a" },
      { next: "planejador", brief: "b" },
      { next: "analista", brief: "c" },
      { next: "done", brief: "fim" },
    ]);
    const result = await runTeam("x", deps);
    assert.equal(calls.analista.length, 2);
    assert.equal(shape(result.trace).filter((item) => item === "handoff→analista").length, 2);
  });

  test("proposta do executor encerra a equipe sem nova decisão (US2)", async () => {
    const { deps, calls } = fakeTeam(
      [
        { next: "analista", brief: "a" },
        { next: "planejador", brief: "b" },
        { next: "executor", brief: "abra o incidente" },
        { next: "analista", brief: "nunca deveria chegar aqui" },
      ],
      {
        async runExecutor(input) {
          calls.executor.push(input);
          return {
            trace: [
              { type: "action", at: 0, tool: "open_incident", args: { title: "x", service: "checkout-api", severity: "high" } },
              { type: "observation", at: 1, result: { status: "awaiting_approval", message: "…" } },
            ],
            summary: "",
          };
        },
      },
    );

    const result = await runTeam("x", deps);

    assert.equal(calls.decide, 3, "o supervisor não é chamado depois da proposta");
    assert.equal(result.proposed, true);
    assert.equal(result.answer, PROPOSAL_ANSWER);
    assert.equal(result.blackboard.outcomes.at(-1)?.proposal, true);
    assert.deepEqual(shape(result.trace).slice(-3), ["action@executor", "observation@executor", "handoff→done"]);
  });
});

describe("helpers puros", () => {
  test("tagRole carimba o papel e reindexa sem mutar", () => {
    const input: TraceEvent[] = [{ type: "thought", at: 0, content: "x" }];
    assert.deepEqual(tagRole(input, "planejador", 5), [{ type: "thought", at: 5, content: "x", role: "planejador" }]);
    assert.equal(input[0]?.role, undefined);
  });

  test("hasProposal reconhece a observação da porta em objeto ou string JSON", () => {
    assert.equal(hasProposal([{ type: "observation", at: 0, result: { status: "awaiting_approval" } }]), true);
    assert.equal(hasProposal([{ type: "observation", at: 0, result: '{"status":"awaiting_approval"}' }]), true);
    assert.equal(hasProposal([{ type: "observation", at: 0, result: '{"status":"rejected"}' }]), false);
    assert.equal(hasProposal([{ type: "observation", at: 0, result: "texto" }]), false);
  });
});
