import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { TraceEventSchema, type TraceEvent } from "./api-schemas.ts";
import { formatDuration, LONG_CONTENT_CHARS, summarizeRun, toTraceView } from "./trace-view.ts";

const event = (raw: unknown): TraceEvent => TraceEventSchema.parse(raw);

describe("toTraceView", () => {
  const all = [
    { type: "route", at: 0, node: "roteador", route: "react", reason: "direta", source: "router" },
    { type: "thought", at: 1, node: "react", content: "pensando" },
    { type: "plan", at: 2, node: "planExecute", steps: ["listar", "resumir"] },
    { type: "action", at: 3, node: "react", tool: "list_alerts", args: { status: "firing" } },
    { type: "observation", at: 4, node: "react", result: '[{"id":"A1"}]' },
    { type: "critique", at: 5, node: "reflect", content: "faltou o serviço" },
    { type: "fallback", at: 6, node: "react", from: "modelo-a", to: "modelo-b", reason: "429" },
    { type: "answer", at: 7, node: "resposta", content: "pronto" },
  ];

  test("cada tipo conhecido tem rótulo e ícone próprios, em PT", () => {
    const views = all.map((raw) => toTraceView(event(raw)));
    assert.deepEqual(
      views.map((view) => view.label),
      ["Rota", "Pensamento", "Plano", "Ação", "Observação", "Crítica", "Troca de modelo", "Resposta"],
    );
    assert.equal(new Set(views.map((view) => view.icon)).size, 8);
    assert.deepEqual(
      views.map((view) => view.node),
      ["roteador", "react", "planExecute", "react", "react", "reflect", "react", "resposta"],
    );
  });

  test("plano vira lista de passos na ordem", () => {
    assert.deepEqual(toTraceView(event(all[2])).body, { kind: "steps", steps: ["listar", "resumir"] });
  });

  test("ação mostra a ferramenta e os argumentos indentados", () => {
    assert.deepEqual(toTraceView(event(all[3])).body, {
      kind: "code",
      title: "list_alerts",
      text: JSON.stringify({ status: "firing" }, null, 2),
      long: false,
    });
  });

  test("observação em string JSON é formatada; longa fica recolhida", () => {
    const short = toTraceView(event(all[4])).body;
    assert.equal(short.kind === "code" && short.text, JSON.stringify([{ id: "A1" }], null, 2));

    const big = JSON.stringify(Array.from({ length: 40 }, (_, i) => ({ id: `ALERTA-${i}`, service: "checkout-api" })));
    assert.ok(big.length > LONG_CONTENT_CHARS);
    const long = toTraceView(event({ type: "observation", at: 0, result: big })).body;
    assert.equal(long.kind === "code" && long.long, true);
  });

  test("observação em texto simples continua texto", () => {
    const body = toTraceView(event({ type: "observation", at: 0, result: "sem alertas" })).body;
    assert.equal(body.kind === "code" && body.text, "sem alertas");
  });

  test("troca de modelo mostra origem e destino", () => {
    assert.deepEqual(toTraceView(event(all[6])).body, { kind: "fallback", from: "modelo-a", to: "modelo-b", reason: "429" });
  });

  test("rota traduz a origem da decisão", () => {
    const body = toTraceView(event(all[0])).body;
    assert.equal(body.kind === "route" && body.source, "decidida pelo roteador");
  });

  test("tipo desconhecido aparece de forma genérica com o conteúdo bruto (FR-011)", () => {
    const view = toTraceView(event({ type: "novo", at: 9, foo: 1 }));
    assert.equal(view.kind, "unknown");
    assert.equal(view.label, "Evento: novo");
    assert.equal(view.body.kind === "code" && view.body.text, JSON.stringify({ type: "novo", foo: 1 }, null, 2));
  });

  test("evento sem node fica com node null", () => {
    assert.equal(toTraceView(event({ type: "thought", at: 0, content: "x" })).node, null);
  });
});

describe("summarizeRun", () => {
  test("monta rota e métricas legíveis", () => {
    const summary = summarizeRun({
      requestId: "r1",
      answer: "a",
      trace: [],
      route: { route: "react", reason: "direta", source: "override" },
      metrics: { llmCalls: 2, latencyMs: 1500, promptTokens: 1234, tokenSource: "estimated", modelUsed: "m" },
    });
    assert.deepEqual(summary.route, { route: "react", reason: "direta", source: "escolhida pelo cliente" });
    assert.deepEqual(
      summary.metrics?.map((metric) => metric.value),
      ["2", "1,5 s", "m", "1.234 (estimado)"],
    );
  });

  test("rota e métricas nulas (decisão) são omitidas", () => {
    const summary = summarizeRun({ requestId: "r2", answer: "a", trace: [], route: null, metrics: null });
    assert.equal(summary.route, null);
    assert.equal(summary.metrics, null);
    assert.equal(summary.requestId, "r2");
  });

  test("formatDuration", () => {
    assert.equal(formatDuration(250), "250 ms");
    assert.equal(formatDuration(12_000), "12 s");
  });
});
