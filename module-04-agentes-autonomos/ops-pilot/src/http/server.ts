import { randomUUID } from "node:crypto";

import type { StructuredToolInterface } from "@langchain/core/tools";
import express, { type Express, type NextFunction, type Request, type Response } from "express";
import { z } from "zod";

import { resolveStrategy as resolveStrategyDefault, strategyForRoute } from "../agents/index.ts";
import type { ReasoningStrategy, RouteName } from "../agents/types.ts";
import { loadContextBudget, type ContextBudget } from "../context/context-builder.ts";
import { ChatTimeoutError, ConversationNotFoundError, UnknownStrategyError } from "../domain/errors.ts";
import { buildRequestRecord, chatMetricsOf } from "../domain/request-record.ts";
import {
  computeStats,
  DEFAULT_STATS_WINDOW,
  loadModelPrices,
  parseStatsWindow,
  type ModelPrices,
} from "../domain/request-stats.ts";
import { reflectAndRemember as reflectAndRememberDefault } from "../memory/learning-reflector.ts";
import { createMemoryTools, SqliteMemoryStore, type MemoryStore } from "../memory/memory-store.ts";
import type { ConversationStore } from "../services/conversation-store.repository.ts";
import { createProductionGraph } from "../graph/production-graph.ts";
import { createModelRouter, parseRouteName, type DecideRoute } from "../graph/router.ts";
import { createLogger, errorTypeOf, type Logger } from "../obs/logger.ts";
import { withTimeout } from "../services/chat.service.ts";
import type { RequestStore } from "../services/request-store.repository.ts";
import { SqliteConversationStore } from "../store/sqlite-conversation-store.ts";
import { SqliteRequestStore } from "../store/sqlite-request-store.ts";

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
  /** Onde as execuções do /chat são gravadas (014) — testes usam `:memory:`; padrão `SqliteRequestStore`. */
  requestStore?: RequestStore;
  /** Logger JSON (014) — testes injetam um coletor/silencioso; padrão `createLogger()` (stdout). */
  logger?: Logger;
  /** Relógio — injetável para testes; padrão `() => new Date()`. */
  now?: () => Date;
  /** Preço de prompt por modelo para `GET /stats` — padrão `loadModelPrices(process.env)`. */
  modelPrices?: ModelPrices;
}

const RequestIdParamSchema = z.string().uuid();

/** Gera o id da requisição no servidor (o `X-Request-Id` do cliente é ignorado) e o expõe no cabeçalho. */
function assignRequestId(_req: Request, res: Response, next: NextFunction): void {
  const requestId = randomUUID();
  res.locals.requestId = requestId;
  res.setHeader("X-Request-Id", requestId);
  next();
}

/** Nome da rota do override, ou `null` quando ausente/inválido — só para o log de entrada. */
function overrideForLog(strategy: string | undefined): RouteName | null {
  if (strategy === undefined) {
    return null;
  }
  try {
    return parseRouteName(strategy);
  } catch {
    return null;
  }
}

/**
 * Middleware de erro do Express: traduz falhas em status HTTP, nunca o contrário (Principle III).
 * Todo corpo leva o `requestId` (014); o log é só de metadados — nunca a mensagem nem o stack.
 */
function createErrorMiddleware(logger: Logger, now: () => Date) {
  return (error: unknown, _req: Request, res: Response, _next: NextFunction): void => {
    const requestId = res.locals.requestId as string;
    const startedAt = res.locals.startedAt as Date | undefined;
    const durationMs = startedAt ? now().getTime() - startedAt.getTime() : 0;

    if (error instanceof UnknownStrategyError) {
      logger.log({ event: "request.rejected", requestId, status: 422, errorCode: "unknown_strategy" });
      res.status(422).json({ requestId, error: "unknown_strategy", strategy: error.strategy });
      return;
    }

    if (error instanceof ConversationNotFoundError) {
      logger.log({ event: "request.rejected", requestId, status: 404, errorCode: "conversation_not_found" });
      res.status(404).json({ requestId, error: "conversation_not_found", conversationId: error.conversationId });
      return;
    }

    if (error instanceof ChatTimeoutError) {
      logger.log({ event: "request.failed", requestId, status: 504, errorType: errorTypeOf(error), durationMs });
      res.status(504).json({ requestId, error: "timeout", timeoutMs: error.timeoutMs });
      return;
    }

    logger.log({ event: "request.failed", requestId, status: 500, errorType: errorTypeOf(error), durationMs });
    res.status(500).json({ requestId, error: "internal_error" });
  };
}

export function createApp(options: CreateAppOptions = {}): Express {
  const resolveStrategy = options.resolveStrategy ?? resolveStrategyDefault;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const conversationStore = options.conversationStore ?? new SqliteConversationStore();
  const memoryStore = options.memoryStore ?? new SqliteMemoryStore();
  const reflectAndRemember = options.reflectAndRemember ?? reflectAndRememberDefault;
  const contextBudget = options.contextBudget ?? loadContextBudget(process.env);
  const decideRoute = options.decideRoute ?? createModelRouter();
  const requestStore = options.requestStore ?? new SqliteRequestStore();
  const logger = options.logger ?? createLogger();
  const now = options.now ?? (() => new Date());
  const modelPrices = options.modelPrices ?? loadModelPrices(process.env);

  const app = express();
  app.use(express.json());

  app.post("/chat", assignRequestId, async (req: Request, res: Response, next: NextFunction) => {
    const requestId = res.locals.requestId as string;
    const startedAt = now();
    res.locals.startedAt = startedAt;

    const parsed = ChatRequestSchema.safeParse(req.body);
    if (!parsed.success) {
      logger.log({ event: "request.rejected", requestId, status: 400, errorCode: "invalid_body" });
      res.status(400).json({ requestId, error: "invalid_body", issues: parsed.error.issues });
      return;
    }

    logger.log({
      event: "request.received",
      requestId,
      method: req.method,
      path: req.path,
      hasConversationId: parsed.data.conversationId !== undefined,
      hasUserId: parsed.data.userId !== undefined,
      strategyOverride: overrideForLog(parsed.data.strategy),
    });

    let conversationId: string | undefined;
    // Só execuções que começaram e não terminaram (timeout/erro do grafo) são gravadas aqui; o
    // sucesso é gravado pelo nó `resposta` do grafo (014). Erros antes disso não são persistidos.
    let running = false;
    let abandoned = false;

    try {
      // Override validado antes de qualquer IO: estratégia desconhecida → 422 sem executar nada (FR-010).
      const override = parsed.data.strategy !== undefined ? parseRouteName(parsed.data.strategy) : undefined;

      conversationId = parsed.data.conversationId ?? (await conversationStore.create());
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
        requestStore,
        logger,
        now,
      });
      running = true;
      const result = await withTimeout(
        () =>
          graph.run({
            context: { message: parsed.data.message, window: history, memories: recalled },
            budget: contextBudget,
            override,
            request: { requestId, conversationId: conversationId ?? null, userId, startedAt, abandoned: () => abandoned },
          }),
        timeoutMs,
      );
      running = false;
      const { context: built, route, ...runResult } = result;

      if (userId) {
        void reflectAndRemember(memoryStore, userId, parsed.data.message).catch(() => {});
      }

      await conversationStore.append(conversationId, [
        { role: "user", content: parsed.data.message },
        { role: "assistant", content: result.answer },
      ]);

      res.status(200).json({
        requestId,
        ...runResult,
        route,
        conversationId,
        metrics: chatMetricsOf(runResult.metrics, built),
      });
    } catch (error) {
      if (running) {
        abandoned = true;
        await persistFailure(requestId, conversationId ?? null, parsed.data.userId, startedAt, error);
      }
      next(error);
    }
  });

  /** Grava timeout/erro de execução com trace vazio; falha de gravação só vira log (FR-008). */
  async function persistFailure(
    requestId: string,
    conversationId: string | null,
    userId: string | undefined,
    startedAt: Date,
    error: unknown,
  ): Promise<void> {
    const record = buildRequestRecord({
      requestId,
      conversationId,
      userId,
      startedAt,
      durationMs: now().getTime() - startedAt.getTime(),
      outcome: error instanceof ChatTimeoutError ? "timeout" : "error",
      errorType: errorTypeOf(error),
    });
    try {
      await requestStore.save(record, []);
    } catch (saveError) {
      logger.log({ event: "persistence.failed", requestId, errorType: errorTypeOf(saveError) });
    }
  }

  /** Agregados das execuções gravadas na janela `?since=` (padrão 24h): total, erros, tokens, custo, p50/p95. */
  app.get("/stats", async (req: Request, res: Response, next: NextFunction) => {
    const since = typeof req.query.since === "string" ? req.query.since : DEFAULT_STATS_WINDOW;
    const windowMs = parseStatsWindow(since);
    if (windowMs === undefined) {
      res.status(400).json({ error: "invalid_since", since, hint: "use <n>m, <n>h ou <n>d (máx. 90d), ex.: 24h" });
      return;
    }

    try {
      const to = now();
      const from = new Date(to.getTime() - windowMs);
      const records = await requestStore.listSince(from);
      res.status(200).json({ since, from: from.toISOString(), to: to.toISOString(), ...computeStats(records, modelPrices) });
    } catch (error) {
      next(error);
    }
  });

  app.get("/requests/:id", async (req: Request, res: Response, next: NextFunction) => {
    const id = req.params.id ?? "";
    try {
      const parsedId = RequestIdParamSchema.safeParse(id);
      const found = parsedId.success ? await requestStore.find(parsedId.data) : undefined;
      logger.log({ event: "request.lookup", requestId: id, found: found !== undefined });

      if (!found) {
        res.status(404).json({ error: "request_not_found", requestId: id });
        return;
      }
      res.status(200).json(found);
    } catch (error) {
      next(error);
    }
  });

  app.use(createErrorMiddleware(logger, now));

  return app;
}
