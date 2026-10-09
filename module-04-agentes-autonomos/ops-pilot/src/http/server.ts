import type { StructuredToolInterface } from "@langchain/core/tools";
import express, { type Express, type NextFunction, type Request, type Response } from "express";
import { z } from "zod";

import { resolveStrategy as resolveStrategyDefault, strategyForRoute } from "../agents/index.ts";
import type { ReasoningStrategy } from "../agents/types.ts";
import { loadContextBudget, type ContextBudget } from "../context/context-builder.ts";
import { ChatTimeoutError, ConversationNotFoundError, UnknownStrategyError } from "../domain/errors.ts";
import { reflectAndRemember as reflectAndRememberDefault } from "../memory/learning-reflector.ts";
import { createMemoryTools, SqliteMemoryStore, type MemoryStore } from "../memory/memory-store.ts";
import type { ConversationStore } from "../services/conversation-store.repository.ts";
import { createProductionGraph } from "../graph/production-graph.ts";
import { createModelRouter, parseRouteName, type DecideRoute } from "../graph/router.ts";
import { withTimeout } from "../services/chat.service.ts";
import { SqliteConversationStore } from "../store/sqlite-conversation-store.ts";

const DEFAULT_TIMEOUT_MS = 180_000;

/** Quantidade máxima de mensagens de histórico consideradas na composição do prompt (FR-004). */
const HISTORY_LIMIT = 12;

/** Quantidade máxima de fatos recuperados por pergunta (spec 007, FR-005). */
const RECALL_LIMIT = 3;

export const ChatRequestSchema = z.object({
  message: z.string().min(1, "message é obrigatório e não pode ser vazio"),
  strategy: z.string().optional(),
  reflect: z.boolean().optional(),
  conversationId: z.string().optional(),
  userId: z.string().optional(),
});

export type ChatRequestBody = z.infer<typeof ChatRequestSchema>;

export interface CreateAppOptions {
  /** Sobrescreve a resolução nome->estratégia — usado por testes para injetar fakes, sem rede. */
  resolveStrategy?: (
    name: string | undefined,
    reflect: boolean | undefined,
    extraTools?: StructuredToolInterface[],
  ) => ReasoningStrategy;
  /** Teto de tempo por requisição, em ms. Padrão 180000 (FR-008); overridable para testes rápidos. */
  timeoutMs?: number;
  /** Sobrescreve o armazenamento de conversas — usado por testes para injetar fakes, sem SQLite real. */
  conversationStore?: ConversationStore;
  /** Sobrescreve o armazenamento de memória semântica — usado por testes para injetar fakes, sem SQLite/modelo real. */
  memoryStore?: MemoryStore;
  /** Sobrescreve o refletor de aprendizado (008) — usado por testes para injetar fakes, sem rede. */
  reflectAndRemember?: typeof reflectAndRememberDefault;
  /** Sobrescreve os tetos de contexto — usado por testes; padrão `loadContextBudget(process.env)`. */
  contextBudget?: ContextBudget;
  /** Sobrescreve o roteador do grafo de produção (012) — usado por testes; padrão `createModelRouter()`. */
  decideRoute?: DecideRoute;
}

/** Middleware de erro do Express: traduz falhas em status HTTP, nunca o contrário (Principle III). */
function errorMiddleware(error: unknown, _req: Request, res: Response, _next: NextFunction): void {
  if (error instanceof UnknownStrategyError) {
    res.status(422).json({ error: "unknown_strategy", strategy: error.strategy });
    return;
  }

  if (error instanceof ConversationNotFoundError) {
    res.status(404).json({ error: "conversation_not_found", conversationId: error.conversationId });
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
  const conversationStore = options.conversationStore ?? new SqliteConversationStore();
  const memoryStore = options.memoryStore ?? new SqliteMemoryStore();
  const reflectAndRemember = options.reflectAndRemember ?? reflectAndRememberDefault;
  const contextBudget = options.contextBudget ?? loadContextBudget(process.env);
  const decideRoute = options.decideRoute ?? createModelRouter();

  const app = express();
  app.use(express.json());

  app.post("/chat", async (req: Request, res: Response, next: NextFunction) => {
    const parsed = ChatRequestSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "invalid_body", issues: parsed.error.issues });
      return;
    }

    try {
      // Override validado antes de qualquer IO: estratégia desconhecida → 422 sem executar nada (FR-010).
      const override = parsed.data.strategy !== undefined ? parseRouteName(parsed.data.strategy) : undefined;

      const conversationId = parsed.data.conversationId ?? (await conversationStore.create());
      const history = parsed.data.conversationId
        ? await conversationStore.lastMessages(conversationId, HISTORY_LIMIT)
        : [];

      const userId = parsed.data.userId;
      const recalled = userId
        ? await memoryStore.recall(userId, parsed.data.message, RECALL_LIMIT)
        : [];
      const extraTools = userId ? createMemoryTools(memoryStore, userId) : undefined;

      const graph = createProductionGraph({
        decideRoute,
        strategyFor: (route) => strategyForRoute(route, parsed.data.reflect, extraTools, resolveStrategy),
      });
      const result = await withTimeout(
        () =>
          graph.run({
            context: { message: parsed.data.message, window: history, memories: recalled },
            budget: contextBudget,
            override,
          }),
        timeoutMs,
      );
      const { context: built, route, ...runResult } = result;

      if (userId) {
        void reflectAndRemember(memoryStore, userId, parsed.data.message).catch(() => {});
      }

      await conversationStore.append(conversationId, [
        { role: "user", content: parsed.data.message },
        { role: "assistant", content: result.answer },
      ]);

      res.status(200).json({
        ...runResult,
        route,
        conversationId,
        metrics: {
          ...runResult.metrics,
          historyMessages: built.window.length,
          contextBreakdown: built.breakdown,
          contextTrimmed: built.trimmed,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  app.use(errorMiddleware);

  return app;
}
