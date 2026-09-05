import { ChatTimeoutError } from "../domain/errors.ts";
import type { ReasoningStrategy, RunOptions, RunResult } from "../agents/types.ts";

/**
 * Executa `strategy.run(...)` contra um teto de tempo. Se `timeoutMs` for atingido primeiro,
 * rejeita com `ChatTimeoutError` — a execução da estratégia em si não é cancelada (research.md
 * item 2), apenas deixa de ser aguardada pelo chamador. Sempre limpa o timer interno.
 */
export function runWithTimeout(
  strategy: ReasoningStrategy,
  input: string,
  options: RunOptions | undefined,
  timeoutMs: number,
): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new ChatTimeoutError(timeoutMs));
    }, timeoutMs);

    strategy.run(input, options).then(
      (result) => {
        clearTimeout(timer);
        resolve(result);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}
