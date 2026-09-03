import { z } from "zod";

import { planAndExecuteStrategy } from "./agents/plan-and-execute.ts";
import { reactStrategy } from "./agents/react.ts";
import { formatMetrics, formatTrace } from "./agents/trace.ts";
import type { ReasoningStrategy, RunOptions } from "./agents/types.ts";
import type { Incident, OpsState } from "./domain/ops-store.ts";
import { store } from "./services/ops-store.memory.ts";

const SCENARIO_IDS = ["C1", "C2", "C3"] as const;
type ScenarioId = (typeof SCENARIO_IDS)[number];

const STRATEGY_NAMES = ["react", "plan-and-execute"] as const;
type StrategyName = (typeof STRATEGY_NAMES)[number];

const STRATEGIES: Record<StrategyName, ReasoningStrategy> = {
  react: reactStrategy,
  "plan-and-execute": planAndExecuteStrategy,
};

const NUMBER_WORDS_PT = [
  "zero",
  "um",
  "dois",
  "três",
  "quatro",
  "cinco",
  "seis",
  "sete",
  "oito",
  "nove",
  "dez",
];

/** Verifica se `answer` menciona a contagem `n`, em dígito (com limite de palavra) ou por extenso. */
export function answerMentionsCount(answer: string, n: number): boolean {
  const normalized = answer.toLowerCase();
  if (new RegExp(`(?<![0-9])${n}(?![0-9])`).test(normalized)) {
    return true;
  }
  const word = NUMBER_WORDS_PT[n];
  return word !== undefined && normalized.includes(word);
}

export function firingAlerts(state: OpsState) {
  return state.alerts.filter((alert) => alert.status === "firing");
}

/** Incidentes presentes em `after` mas não em `before` — os abertos durante a execução avaliada. */
export function newIncidents(before: OpsState, after: OpsState): Incident[] {
  const beforeIds = new Set(before.incidents.map((incident) => incident.id));
  return after.incidents
    .filter((incident) => !beforeIds.has(incident.id))
    .sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime());
}

function serviceNameOf(state: OpsState, serviceId: string): string | undefined {
  return state.services.find((service) => service.id === serviceId)?.name;
}

export interface CheckResult {
  pass: boolean;
  reason: string;
}

function pass(reason: string): CheckResult {
  return { pass: true, reason };
}

function fail(reason: string): CheckResult {
  return { pass: false, reason };
}

export interface Scenario {
  readonly id: ScenarioId;
  readonly input: string;
  check(before: OpsState, after: OpsState, answer: string): CheckResult;
}

/**
 * C1 (direto): alertas não têm campo de severidade no domínio (só incidentes têm) — "críticos" não
 * filtra nada real. A resposta correta é a contagem total de firing; isso testa se a estratégia
 * alucina um filtro inexistente em vez de responder com o que o store realmente tem.
 */
function checkC1(before: OpsState, after: OpsState, answer: string): CheckResult {
  const expected = firingAlerts(before).length;
  const stateUnchanged =
    after.incidents.length === before.incidents.length && after.alerts.length === before.alerts.length;

  if (!stateUnchanged) {
    return fail("pergunta somente leitura, mas o estado do store foi alterado");
  }
  if (!answerMentionsCount(answer, expected)) {
    return fail(`resposta não menciona a contagem esperada de alertas firing (${expected})`);
  }
  return pass(`contagem correta (${expected}) e store inalterado`);
}

/**
 * C2 (estruturado): "catalog" não é um serviço real do seed (5 serviços: checkout-api, payments-api,
 * inventory-service, notifications-service, auth-service), então nenhum incidente real pode existir
 * para ele — o check não exige isso. "sev2" é ambíguo frente ao enum low/medium/high/critical, então
 * a severidade não é conferida; o que importa é a ordem (checkout antes de payments) e que só o
 * primeiro incidente aberto (checkout) termine resolvido.
 */
function checkC2(before: OpsState, after: OpsState): CheckResult {
  const created = newIncidents(before, after);

  const checkoutIncident = created.find((incident) => serviceNameOf(after, incident.serviceId) === "checkout-api");
  const paymentsIncident = created.find((incident) => serviceNameOf(after, incident.serviceId) === "payments-api");

  if (!checkoutIncident) {
    return fail("nenhum incidente foi aberto para checkout-api");
  }
  if (!paymentsIncident) {
    return fail("nenhum incidente foi aberto para payments-api");
  }
  if (new Date(checkoutIncident.createdAt).getTime() > new Date(paymentsIncident.createdAt).getTime()) {
    return fail("o incidente de checkout não foi aberto antes do de payments");
  }
  if (checkoutIncident.status !== "resolved") {
    return fail("o primeiro incidente aberto (checkout) não foi resolvido");
  }
  if (paymentsIncident.status === "resolved") {
    return fail("payments foi resolvido, mas apenas o primeiro incidente deveria ser");
  }

  return pass("incidentes de checkout (resolved) e payments (open) abertos na ordem correta");
}

/**
 * C3 (dinâmico): no seed, os 3 alertas firing têm o mesmo createdAt — não há um "mais antigo" único.
 * Por isso o check aceita qualquer serviço com alerta firing como alvo válido, e confere apenas que
 * exatamente um incidente novo foi aberto e que a contagem restante (firing - 1) foi reportada.
 */
function checkC3(before: OpsState, after: OpsState, answer: string): CheckResult {
  const firingBefore = firingAlerts(before);
  const created = newIncidents(before, after);

  if (created.length !== 1) {
    return fail(`esperava exatamente 1 incidente novo, encontrou ${created.length}`);
  }

  const [incident] = created;
  const targetsFiringService = firingBefore.some((alert) => alert.serviceId === incident?.serviceId);
  if (!targetsFiringService) {
    return fail("o incidente aberto não corresponde a nenhum serviço com alerta firing");
  }

  const expectedRemaining = firingBefore.length - 1;
  if (!answerMentionsCount(answer, expectedRemaining)) {
    return fail(`resposta não menciona quantos alertas restaram (${expectedRemaining})`);
  }

  return pass(`incidente aberto para um serviço com alerta firing e contagem restante (${expectedRemaining}) reportada`);
}

export const SCENARIOS: readonly Scenario[] = [
  { id: "C1", input: "Quantos alertas críticos estão disparando?", check: checkC1 },
  {
    id: "C2",
    input: "Abra três incidentes sev2 para checkout, payment e catalog, nessa mesma ordem, e resolva o primeiro.",
    check: checkC2,
  },
  {
    id: "C3",
    input: "Dos alertas disparando, abra um incidente para o mais antigo e diga quantos sobraram.",
    check: checkC3,
  },
];

export interface BenchRow {
  scenario: ScenarioId;
  strategy: StrategyName;
  pass: boolean;
  reason: string;
  llmCalls: number;
  latencyMs: number;
}

const SUMMARY_COLUMNS = ["cenário", "estratégia", "acerto", "llmCalls", "latencyMs"] as const;

/** Formata a tabela consolidada como texto alinhado em colunas. Função pura, determinística. */
export function formatSummaryTable(rows: readonly BenchRow[]): string {
  const cells = rows.map((row) => [
    row.scenario,
    row.strategy,
    row.pass ? "OK" : "FALHOU",
    String(row.llmCalls),
    String(row.latencyMs),
  ]);

  const widths = SUMMARY_COLUMNS.map((header, col) =>
    Math.max(header.length, ...cells.map((row) => row[col]?.length ?? 0)),
  );

  const formatRow = (values: readonly string[]): string =>
    values.map((value, col) => value.padEnd(widths[col] ?? 0)).join("  ");

  return [
    formatRow(SUMMARY_COLUMNS),
    formatRow(widths.map((width) => "-".repeat(width))),
    ...cells.map(formatRow),
  ].join("\n");
}

const ArgsSchema = z.object({
  scenario: z.enum(SCENARIO_IDS).optional(),
  noReplanner: z.boolean(),
  maxIterations: z.number().int().positive(),
});

type Args = z.infer<typeof ArgsSchema>;

export function parseArgs(argv: string[]): Args {
  let scenario: string | undefined;
  let noReplanner = false;
  let maxIterations = 8;

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];

    if (flag === "--scenario") {
      scenario = value?.trim().toUpperCase();
      i += 1;
    } else if (flag === "--no-replanner") {
      noReplanner = true;
    } else if (flag === "--max-iterations") {
      maxIterations = Number(value);
      i += 1;
    }
  }

  const result = ArgsSchema.safeParse({ scenario, noReplanner, maxIterations });
  if (!result.success) {
    console.error("Argumentos inválidos:", result.error.flatten().fieldErrors);
    process.exit(1);
  }

  return result.data;
}

async function runOne(scenario: Scenario, strategyName: StrategyName, args: Args): Promise<BenchRow> {
  store.reset();
  const before = store.getState();

  const options: RunOptions = { maxIterations: args.maxIterations };
  if (strategyName === "plan-and-execute") {
    options.noReplanner = args.noReplanner;
  }

  const result = await STRATEGIES[strategyName].run(scenario.input, options);
  const after = store.getState();
  const { pass: passed, reason } = scenario.check(before, after, result.answer);

  console.log(`\n=== ${scenario.id} · ${strategyName} ===`);
  console.log(formatTrace(result.trace));
  console.log(`\nresposta: ${result.answer}`);
  console.log(formatMetrics(result.metrics));
  console.log(`acerto: ${passed ? "OK" : "FALHOU"} — ${reason}`);

  return {
    scenario: scenario.id,
    strategy: strategyName,
    pass: passed,
    reason,
    llmCalls: result.metrics.llmCalls,
    latencyMs: result.metrics.latencyMs,
  };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const scenarios = SCENARIOS.filter((scenario) => !args.scenario || scenario.id === args.scenario);

  const rows: BenchRow[] = [];
  for (const scenario of scenarios) {
    for (const strategyName of STRATEGY_NAMES) {
      rows.push(await runOne(scenario, strategyName, args));
    }
  }

  console.log("\n=== tabela consolidada ===");
  console.log(formatSummaryTable(rows));
}

const isMainModule = process.argv[1] !== undefined && import.meta.url === `file://${process.argv[1]}`;
if (isMainModule) {
  main().catch((error: unknown) => {
    console.error("Falha ao rodar o bench:", error);
    process.exitCode = 1;
  });
}
