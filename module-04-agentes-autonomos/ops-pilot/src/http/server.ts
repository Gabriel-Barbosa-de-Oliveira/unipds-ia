import express, { type Express, type NextFunction, type Request, type Response } from "express";
import { z } from "zod";

import { resolveStrategy as resolveStrategyDefault } from "../agents/index.ts";
import type { ReasoningStrategy } from "../agents/types.ts";
import { ChatTimeoutError, UnknownStrategyError } from "../domain/errors.ts";
import { runWithTimeout } from "../services/chat.service.ts";

const DEFAULT_TIMEOUT_MS = 180_000;

export const ChatRequestSchema = z.object({
  message: z.string().min(1, "message é obrigatório e não pode ser vazio"),
  strategy: z.string().optional(),
  reflect: z.boolean().optional(),
});

export type ChatRequestBody = z.infer<typeof ChatRequestSchema>;

export interface CreateAppOptions {
  /** Sobrescreve a resolução nome->estratégia — usado por testes para injetar fakes, sem rede. */
  resolveStrategy?: (name: string | undefined, reflect: boolean | undefined) => ReasoningStrategy;
  /** Teto de tempo por requisição, em ms. Padrão 180000 (FR-008); overridable para testes rápidos. */
  timeoutMs?: number;
}

/** Middleware de erro do Express: traduz falhas em status HTTP, nunca o contrário (Principle III). */
function errorMiddleware(error: unknown, _req: Request, res: Response, _next: NextFunction): void {
  if (error instanceof UnknownStrategyError) {
    res.status(422).json({ error: "unknown_strategy", strategy: error.strategy });
    return;
  }

  if (error instanceof ChatTimeoutError) {
    res.status(504).json({ error: "timeout", timeoutMs: error.timeoutMs });
    return;
  }

  console.error("Erro inesperado no /chat:", error);
  res.status(500).json({ error: "internal_error" });
}

export function createApp(options: CreateAppOptions = {}): Express {
  const resolveStrategy = options.resolveStrategy ?? resolveStrategyDefault;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  const app = express();
  app.use(express.json());

  app.post("/chat", async (req: Request, res: Response, next: NextFunction) => {
    const parsed = ChatRequestSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "invalid_body", issues: parsed.error.issues });
      return;
    }

    try {
      const strategy = resolveStrategy(parsed.data.strategy, parsed.data.reflect);
      const result = await runWithTimeout(strategy, parsed.data.message, undefined, timeoutMs);
      res.status(200).json(result);
    } catch (error) {
      next(error);
    }
  });

  app.use(errorMiddleware);

  return app;
}
