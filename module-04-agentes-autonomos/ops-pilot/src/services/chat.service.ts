import { ChatTimeoutError } from "../domain/errors.ts";
import type { ReasoningStrategy, RunOptions, RunResult } from "../agents/types.ts";

/**
 * Executa `run()` contra um teto de tempo. Se `timeoutMs` for atingido primeiro, rejeita com
 * `ChatTimeoutError` — a execução em si não é cancelada (research.md item 2 da 003), apenas deixa
 * de ser aguardada pelo chamador. Sempre limpa o timer interno.
 */
export function withTimeout<T>(run: () => Promise<T>, timeoutMs: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new ChatTimeoutError(timeoutMs));
    }, timeoutMs);

    run().then(
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

/** Executa `strategy.run(...)` contra um teto de tempo (ver `withTimeout`). */
export function runWithTimeout(
  strategy: ReasoningStrategy,
  input: string,
  options: RunOptions | undefined,
  timeoutMs: number,
): Promise<RunResult> {
  return withTimeout(() => strategy.run(input, options), timeoutMs);
}
