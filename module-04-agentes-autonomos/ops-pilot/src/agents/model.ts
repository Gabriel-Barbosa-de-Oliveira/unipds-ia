import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { AIMessageChunk, BaseMessage } from "@langchain/core/messages";
import { RunnableBinding, type Runnable, type RunnableConfig } from "@langchain/core/runnables";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

/** Tentativas por modelo (1 + 1 retry) — vale para o principal e para a reserva (spec 013). */
export const DEFAULT_STOP_AFTER_ATTEMPT = 2;

export interface ModelConfig {
  primary: string;
  /** Modelo de reserva; ausente quando `OPENROUTER_MODEL_FALLBACK` está vazio ou igual ao principal. */
  fallback?: string;
}

const modelConfigSchema = z.object({
  OPENROUTER_MODEL: z.string({ required_error: "OPENROUTER_MODEL não configurada" }).trim().min(1, "OPENROUTER_MODEL não configurada"),
  OPENROUTER_MODEL_FALLBACK: z.string().trim().optional(),
});

/**
 * Lê os modelos principal e de reserva do `env` recebido (nunca do `.env` direto — FR-012).
 * Pura. Reserva vazia ou igual ao principal significa "sem reserva" (FR-005).
 */
export function loadModelConfig(env: Readonly<Record<string, string | undefined>>): ModelConfig {
  const parsed = modelConfigSchema.safeParse(env);
  if (!parsed.success) {
    throw new Error("OPENROUTER_MODEL não configurada");
  }

  const primary = parsed.data.OPENROUTER_MODEL;
  const fallback = parsed.data.OPENROUTER_MODEL_FALLBACK;
  return fallback && fallback !== primary ? { primary, fallback } : { primary };
}

const TRANSIENT_STATUS = new Set([408, 429]);
const TRANSIENT_CODES = new Set(["ECONNRESET", "ETIMEDOUT", "ECONNREFUSED", "EAI_AGAIN", "EPIPE"]);
const TRANSIENT_NAMES = new Set(["APIConnectionError", "APIConnectionTimeoutError", "TimeoutError", "AbortError"]);
const TRANSIENT_MESSAGE = /fetch failed|rate limit|timed? ?out|socket hang up/i;

function statusOf(error: Record<string, unknown>): number | undefined {
  const response = error.response as Record<string, unknown> | undefined;
  const status = error.status ?? response?.status;
  return typeof status === "number" ? status : undefined;
}

/**
 * Classifica uma falha de chamada ao modelo: true só para o que vale tentar de novo no mesmo
 * modelo (limite de requisições, erro do provedor, rede, timeout — FR-002/FR-003). Pura.
 */
export function isTransientModelError(error: unknown): boolean {
  if (typeof error !== "object" || error === null) {
    return false;
  }

  const record = error as Record<string, unknown>;
  const status = statusOf(record);
  if (status !== undefined) {
    return TRANSIENT_STATUS.has(status) || status >= 500;
  }

  const code = typeof record.code === "string" ? record.code : "";
  const name = typeof record.name === "string" ? record.name : "";
  const message = typeof record.message === "string" ? record.message : "";

  return TRANSIENT_CODES.has(code) || code.startsWith("UND_ERR_") || TRANSIENT_NAMES.has(name) || TRANSIENT_MESSAGE.test(message);
}

/**
 * `ChatOpenAI` sobre OpenRouter para um modelo específico. `maxRetries: 0` desliga o retry interno
 * do cliente HTTP — a única política de retry é a do `withRetry` de `createModel` (research.md item 2).
 */
export function baseModel(modelId: string): ChatOpenAI {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY não configurada");
  }

  return new ChatOpenAI({
    model: modelId,
    apiKey,
    configuration: { baseURL: OPENROUTER_BASE_URL },
    temperature: 0,
    maxRetries: 0,
  });
}

export interface ModelDeps {
  /** Fábrica de modelo por id — testes injetam fakes, sem rede. Padrão `baseModel`. */
  baseModel?: (modelId: string) => BaseChatModel;
  stopAfterAttempt?: number;
}

type ModelInput = BaseLanguageModelInput;

/** Interrompe o retry do `withRetry` quando a falha não é transitória (o p-retry aborta se o callback lança). */
function stopOnPermanentError(error: unknown): void {
  if (!isTransientModelError(error)) {
    throw error;
  }
}

/**
 * Fábrica única de modelo resiliente (spec 013): `build(principal).withRetry()` com
 * `.withFallbacks([build(reserva).withRetry()])` quando há reserva configurada.
 *
 * `build` especializa cada ramo ANTES da composição (`withStructuredOutput`, `bindTools`), porque
 * o runnable devolvido por `withFallbacks` não tem esses métodos (research.md item 3). Sem `build`,
 * devolve o modelo de chat puro (mensagens → mensagem).
 */
export function createModel<T = BaseMessage>(
  build: (model: BaseChatModel) => Runnable<ModelInput, T> = (model) => model as unknown as Runnable<ModelInput, T>,
  config: ModelConfig = loadModelConfig(process.env),
  deps: ModelDeps = {},
): Runnable<ModelInput, T> {
  const makeModel = deps.baseModel ?? baseModel;
  const stopAfterAttempt = deps.stopAfterAttempt ?? DEFAULT_STOP_AFTER_ATTEMPT;
  const withRetry = (modelId: string) =>
    build(makeModel(modelId)).withRetry({ stopAfterAttempt, onFailedAttempt: stopOnPermanentError });

  const primary = withRetry(config.primary);
  if (!config.fallback) {
    return primary;
  }

  const backup = withRetry(config.fallback);
  return primary.withFallbacks([backup]);
}

/**
 * Modelo resiliente com tools já ligadas, aceito por `createReactAgent`. O LangGraph 0.2.74
 * (`_shouldBindTools`) chama `llm.bindTools(...)` a menos que receba um `RunnableBinding` com as
 * mesmas tools em `kwargs.tools` — e o runnable de `withFallbacks` não tem `bindTools`. Por isso o
 * resultado é embrulhado num binding que só declara as tools (idênticas às de cada ramo), sem
 * alterar a chamada (research.md item 4).
 */
export function toolCallingModel(
  tools: StructuredToolInterface[],
  config: ModelConfig = loadModelConfig(process.env),
  deps: ModelDeps = {},
): RunnableBinding<ModelInput, AIMessageChunk> {
  const bindTools = (model: BaseChatModel) => {
    if (!model.bindTools) {
      throw new Error(`O modelo ${model._llmType()} não suporta tool calling`);
    }
    return model.bindTools(tools) as Runnable<ModelInput, AIMessageChunk>;
  };

  const probe = bindTools((deps.baseModel ?? baseModel)(config.primary));
  const toolSpecs: unknown = RunnableBinding.isRunnableBinding(probe) ? probe.kwargs?.tools : undefined;

  return new RunnableBinding({
    bound: createModel(bindTools, config, deps),
    // `tools` é opção de chamada do chat model, não de RunnableConfig — o tipo genérico não a conhece.
    kwargs: { tools: toolSpecs } as Partial<RunnableConfig>,
    config: {},
  });
}
