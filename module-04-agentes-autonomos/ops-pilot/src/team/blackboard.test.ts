import assert from "node:assert/strict";
import { test } from "node:test";

import {
  addFacts,
  addHandoff,
  addOutcome,
  createBlackboard,
  fallbackAnswer,
  renderBlackboard,
  setPlan,
} from "./blackboard.ts";

test("createBlackboard começa só com o pedido", () => {
  assert.deepEqual(createBlackboard("o checkout está lento"), {
    request: "o checkout está lento",
    facts: [],
    plan: null,
    outcomes: [],
    handoffs: [],
  });
});

test("addFacts acrescenta sem mutar e ignora duplicados", () => {
  const empty = createBlackboard("x");
  const one = addFacts(empty, [{ statement: "alert-1 firing (critical)", source: "list_alerts" }]);
  const again = addFacts(one, [
    { statement: "alert-1 firing (critical)", source: "list_alerts" },
    { statement: "sem incidentes abertos", source: "list_incidents" },
  ]);
  assert.equal(empty.facts.length, 0);
  assert.equal(one.facts.length, 1);
  assert.deepEqual(
    again.facts.map((fact) => fact.source),
    ["list_alerts", "list_incidents"],
  );
});

test("setPlan substitui; addOutcome e addHandoff acrescentam", () => {
  let board = setPlan(createBlackboard("x"), ["a", "b"]);
  board = setPlan(board, ["c"]);
  board = addOutcome(board, { summary: "proposto", proposal: true });
  board = addHandoff(board, { from: "supervisor", to: "analista", brief: "levante" });
  assert.deepEqual(board.plan, ["c"]);
  assert.equal(board.outcomes.length, 1);
  assert.equal(board.handoffs[0]?.to, "analista");
});

test("renderBlackboard mostra pedido, fatos com origem, plano, resultados e passagens", () => {
  let board = createBlackboard("investigue o checkout");
  board = addFacts(board, [{ statement: "alert-1 firing", source: "list_alerts" }]);
  board = setPlan(board, ["abrir incidente"]);
  board = addOutcome(board, { summary: "aguardando aprovação", proposal: true });
  board = addHandoff(board, { from: "supervisor", to: "planejador", brief: "planeje" });
  const text = renderBlackboard(board);
  for (const piece of ["investigue o checkout", "alert-1 firing", "list_alerts", "1. abrir incidente", "aguardando aprovação", "planejador"]) {
    assert.ok(text.includes(piece), piece);
  }
});

test("renderBlackboard limita a 20 fatos e trunca itens longos", () => {
  const facts = Array.from({ length: 25 }, (_, index) => ({ statement: `fato-${index}`, source: "pedido" as const }));
  let board = addFacts(createBlackboard("x"), facts);
  board = addFacts(board, [{ statement: "y".repeat(800), source: "pedido" }]);
  const text = renderBlackboard(board);
  assert.equal(text.includes("fato-0 "), false);
  assert.ok(text.includes("fato-24"));
  assert.equal(text.includes("y".repeat(501)), false);
});

test("fallbackAnswer cita o motivo e resume o quadro, inclusive vazio", () => {
  let board = createBlackboard("x");
  assert.match(fallbackAnswer(board, "teto de 6 passagens atingido"), /teto de 6 passagens atingido/);
  board = addFacts(board, [{ statement: "alert-1 firing", source: "list_alerts" }]);
  board = setPlan(board, ["abrir incidente"]);
  const answer = fallbackAnswer(board, "decisão inválida");
  assert.match(answer, /alert-1 firing/);
  assert.match(answer, /abrir incidente/);
});
