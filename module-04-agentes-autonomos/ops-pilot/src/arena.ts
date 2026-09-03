import { z } from "zod";

import { planAndExecuteStrategy } from "./agents/plan-and-execute.ts";
import { reactStrategy } from "./agents/react.ts";
import { withReflection } from "./agents/reflection.ts";
import { formatMetrics, formatTrace } from "./agents/trace.ts";
import type { ReasoningStrategy } from "./agents/types.ts";

const STRATEGY_NAMES = [
  "react",
  "plan-and-execute",
  "reflect:react",
  "reflect:plan-and-execute",
] as const;

const ArgsSchema = z.object({
  input: z.string().min(1, "--input é obrigatório e não pode ser vazio"),
  strategies: z.array(z.enum(STRATEGY_NAMES)).min(1),
  maxIterations: z.number().int().positive(),
});

type Args = z.infer<typeof ArgsSchema>;

/** Registro de estratégias disponíveis para a arena. */
const STRATEGIES: Record<(typeof STRATEGY_NAMES)[number], ReasoningStrategy> = {
  react: reactStrategy,
  "plan-and-execute": planAndExecuteStrategy,
  "reflect:react": withReflection(reactStrategy),
  "reflect:plan-and-execute": withReflection(planAndExecuteStrategy),
};

function parseArgs(argv: string[]): Args {
  let input: string | undefined;
  let strategies: string[] | undefined;
  let maxIterations = 8;

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];

    if (flag === "--input") {
      input = value;
      i += 1;
    } else if (flag === "--strategies") {
      strategies = value?.split(",").map((s) => s.trim());
      i += 1;
    } else if (flag === "--max-iterations") {
      maxIterations = Number(value);
      i += 1;
    }
  }

  const result = ArgsSchema.safeParse({
    input,
    strategies: strategies ?? [...STRATEGY_NAMES],
    maxIterations,
  });

  if (!result.success) {
    console.error("Argumentos inválidos:", result.error.flatten().fieldErrors);
    process.exit(1);
  }

  return result.data;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  const runs: { name: string; llmCalls: number; latencyMs: number }[] = [];

  for (const name of args.strategies) {
    const strategy = STRATEGIES[name];
    const result = await strategy.run(args.input, { maxIterations: args.maxIterations });

    console.log(`\n=== ${name} ===`);
    console.log(formatTrace(result.trace));
    console.log(`\nresposta: ${result.answer}`);
    console.log(formatMetrics(result.metrics));

    runs.push({ name, ...result.metrics });
  }

  if (runs.length > 1) {
    console.log("\n=== resumo comparativo ===");
    for (const run of runs) {
      console.log(`${run.name}: llmCalls=${run.llmCalls} latencyMs=${run.latencyMs}`);
    }
  }
}

main().catch((error: unknown) => {
  console.error("Falha ao rodar a arena:", error);
  process.exitCode = 1;
});
