import assert from "node:assert/strict";
import { after, before, describe, test } from "node:test";

import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { RunnableBinding } from "@langchain/core/runnables";
import { tool, type StructuredToolInterface } from "@langchain/core/tools";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

import { createModel, isTransientModelError, loadModelConfig, toolCallingModel } from "./model.ts";
import { ModelUsageTracker, summarizeModelUsage } from "./model-usage.ts";

function httpError(status: number, message = `HTTP ${status}`): Error {
  return Object.assign(new Error(message), { status });
}

type Step = "ok" | Error;

/** Chat model fake que segue um roteiro de falhas (o último passo se repete), sem rede. */
class ScriptedChatModel extends BaseChatModel {
  calls = 0;

  constructor(
    readonly modelId: string,
    private readonly script: Step[],
  ) {
    super({});
  }

  _llmType(): string {
    return "scripted";
  }

  override invocationParams(): Record<string, unknown> {
    return { model: this.modelId };
  }

  // Mesmo formato do ChatOpenAI.bindTools: tools em `kwargs`, que é o que o createReactAgent inspeciona.
  override bindTools(tools: StructuredToolInterface[]) {
    return new RunnableBinding({
      bound: this,
      kwargs: { tools: tools.map((t) => ({ type: "function", function: { name: t.name } })) } as never,
      config: {},
    });
  }

  async _generate(_messages: BaseMessage[]): Promise<ChatResult> {
    const step = this.script[Math.min(this.calls, this.script.length - 1)]!;
    this.calls += 1;
    if (step instanceof Error) {
      throw step;
    }
    const message = new AIMessage(`resposta de ${this.modelId}`);
    return { generations: [{ text: String(message.content), message }] };
  }
}

function fakes(primaryScript: Step[], fallbackScript: Step[] = ["ok"]) {
  const models = {
    principal: new ScriptedChatModel("principal", primaryScript),
    reserva: new ScriptedChatModel("reserva", fallbackScript),
  };
  const baseModel = (id: string) => models[id as keyof typeof models];
  return { models, deps: { baseModel } };
}

const WITH_FALLBACK = { primary: "principal", fallback: "reserva" };

async function answerOf(promise: Promise<BaseMessage>): Promise<string> {
  return String((await promise).content);
}

describe("loadModelConfig", () => {
  test("lê principal e reserva, com trim", () => {
    assert.deepEqual(loadModelConfig({ OPENROUTER_MODEL: "a" }), { primary: "a" });
    assert.deepEqual(loadModelConfig({ OPENROUTER_MODEL: "a", OPENROUTER_MODEL_FALLBACK: " b " }), {
      primary: "a",
      fallback: "b",
    });
  });

  test("reserva vazia, só espaços ou igual ao principal significa sem reserva", () => {
    for (const fallback of ["", "   ", "a", " a "]) {
      assert.deepEqual(loadModelConfig({ OPENROUTER_MODEL: "a", OPENROUTER_MODEL_FALLBACK: fallback }), { primary: "a" });
    }
  });

  test("sem OPENROUTER_MODEL lança o mesmo erro de antes", () => {
    assert.throws(() => loadModelConfig({}), /OPENROUTER_MODEL não configurada/);
    assert.throws(() => loadModelConfig({ OPENROUTER_MODEL: "  " }), /OPENROUTER_MODEL não configurada/);
  });
});

describe("isTransientModelError", () => {
  test("transitórios: limite de requisições, erro do provedor, rede e timeout", () => {
    const transient: unknown[] = [
      { status: 429 },
      { status: 408 },
      { status: 500 },
      { status: 503 },
      { response: { status: 502 } },
      { code: "ECONNRESET" },
      { code: "ETIMEDOUT" },
      { code: "UND_ERR_SOCKET" },
      { name: "APIConnectionTimeoutError" },
      new Error("fetch failed"),
      new Error("Rate limit exceeded"),
    ];
    for (const error of transient) {
      assert.equal(isTransientModelError(error), true, JSON.stringify(error));
    }
  });

  test("não transitórios: erros do cliente, validação e valores que não são erro", () => {
    const permanent: unknown[] = [{ status: 400 }, { status: 401 }, { status: 404 }, new Error("schema inválido"), "texto", undefined, null];
    for (const error of permanent) {
      assert.equal(isTransientModelError(error), false, String(error));
    }
  });
});

describe("createModel — retry no principal, reserva via withFallbacks (US1)", () => {
  test("caminho feliz: 1 chamada ao principal e nenhuma à reserva", async () => {
    const { models, deps } = fakes(["ok"]);
    assert.equal(await answerOf(createModel(undefined, WITH_FALLBACK, deps).invoke("oi")), "resposta de principal");
    assert.deepEqual([models.principal.calls, models.reserva.calls], [1, 0]);
  });

  test("falha transitória isolada é absorvida pelo retry no principal", async () => {
    const { models, deps } = fakes([httpError(429), "ok"]);
    assert.equal(await answerOf(createModel(undefined, WITH_FALLBACK, deps).invoke("oi")), "resposta de principal");
    assert.deepEqual([models.principal.calls, models.reserva.calls], [2, 0]);
  });

  test("falha transitória persistente esgota as 2 tentativas e cai na reserva", async () => {
    const { models, deps } = fakes([httpError(429)]);
    assert.equal(await answerOf(createModel(undefined, WITH_FALLBACK, deps).invoke("oi")), "resposta de reserva");
    assert.deepEqual([models.principal.calls, models.reserva.calls], [2, 1]);
  });

  test("falha não transitória não é retentada e vai direto para a reserva", async () => {
    const { models, deps } = fakes([httpError(400)]);
    assert.equal(await answerOf(createModel(undefined, WITH_FALLBACK, deps).invoke("oi")), "resposta de reserva");
    assert.deepEqual([models.principal.calls, models.reserva.calls], [1, 1]);
  });

  test("a reserva também tem retry em falha transitória", async () => {
    const { models, deps } = fakes([httpError(400)], [httpError(503), "ok"]);
    assert.equal(await answerOf(createModel(undefined, WITH_FALLBACK, deps).invoke("oi")), "resposta de reserva");
    assert.deepEqual([models.principal.calls, models.reserva.calls], [1, 2]);
  });

  test("sem reserva, o erro propaga depois das tentativas (comportamento de hoje)", async () => {
    const { models, deps } = fakes([httpError(429)]);
    await assert.rejects(createModel(undefined, { primary: "principal" }, deps).invoke("oi"), /HTTP 429/);
    assert.deepEqual([models.principal.calls, models.reserva.calls], [2, 0]);
  });

  test("principal e reserva falhando: a chamada rejeita", async () => {
    const { models, deps } = fakes([httpError(400, "principal quebrado")], [httpError(401, "reserva quebrada")]);
    await assert.rejects(createModel(undefined, WITH_FALLBACK, deps).invoke("oi"));
    assert.deepEqual([models.principal.calls, models.reserva.calls], [1, 1]);
  });

  test("o tracker vê a troca: 1 fallback e modelUsed = reserva (US2)", async () => {
    const { deps } = fakes([httpError(400, "modelo inexistente")]);
    const tracker = new ModelUsageTracker();

    await createModel(undefined, WITH_FALLBACK, deps).invoke("oi", { callbacks: [tracker] });

    assert.deepEqual(summarizeModelUsage(tracker.log, WITH_FALLBACK), {
      fallbacks: [{ from: "principal", to: "reserva", reason: "modelo inexistente" }],
      modelUsed: "reserva",
    });
  });
});

describe("toolCallingModel — compatível com createReactAgent", () => {
  const listAlerts = tool(async () => "[]", {
    name: "list_alerts",
    description: "lista alertas",
    schema: z.object({}),
  });

  test("createReactAgent aceita o modelo resiliente sem religar as tools e usa a reserva quando o principal falha", async () => {
    const { models, deps } = fakes([httpError(400)]);
    const agent = createReactAgent({ llm: toolCallingModel([listAlerts], WITH_FALLBACK, deps), tools: [listAlerts] });

    const result = await agent.invoke({ messages: [{ role: "user", content: "oi" }] });

    assert.equal(String(result.messages.at(-1)?.content), "resposta de reserva");
    assert.deepEqual([models.principal.calls, models.reserva.calls], [1, 1]);
  });
});

describe("configuração pelo ambiente (US3)", () => {
  const saved = { model: process.env.OPENROUTER_MODEL, fallback: process.env.OPENROUTER_MODEL_FALLBACK };

  before(() => {
    process.env.OPENROUTER_MODEL = "principal";
  });

  after(() => {
    for (const [key, value] of [
      ["OPENROUTER_MODEL", saved.model],
      ["OPENROUTER_MODEL_FALLBACK", saved.fallback],
    ] as const) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });

  test("a reserva vem de OPENROUTER_MODEL_FALLBACK, lida a cada montagem (sem mudar código)", async () => {
    process.env.OPENROUTER_MODEL_FALLBACK = "reserva";
    const on = fakes([httpError(400)]);
    assert.equal(await answerOf(createModel(undefined, undefined, on.deps).invoke("oi")), "resposta de reserva");

    delete process.env.OPENROUTER_MODEL_FALLBACK;
    const off = fakes([httpError(400)]);
    await assert.rejects(createModel(undefined, undefined, off.deps).invoke("oi"), /HTTP 400/);
    assert.equal(off.models.reserva.calls, 0);
  });
});
