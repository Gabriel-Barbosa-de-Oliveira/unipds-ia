import { ApiErrorSchema } from "./api-schemas.ts";

export type UiErrorAction = "retry" | "open_settings" | "new_conversation" | "none";

/** Erro pronto para a tela: humano, com próxima ação, sem JSON nem stack (FR-006). */
export interface UiError {
  title: string;
  detail: string;
  requestId?: string;
  action: UiErrorAction;
  /** Status HTTP de origem, quando houve resposta — usado pelo cartão de aprovação. */
  status?: number;
}

export type ErrorInput = { status: number; body: unknown } | { exception: unknown };

export interface ErrorContext {
  /** Origem da war room (`location.origin`), citada na dica de CORS (US5). */
  origin?: string;
}

const TIMEOUT: Omit<UiError, "requestId"> = {
  title: "O copiloto demorou demais para responder",
  detail: "A resposta passou do tempo limite. Tente de novo; se persistir, simplifique a pergunta.",
  action: "retry",
};

function networkError(context: ErrorContext): UiError {
  const where = context.origin ? ` (${context.origin})` : "";
  return {
    title: "Não foi possível falar com a API",
    detail:
      "Confira o endereço da API nas configurações e se esta origem" +
      `${where} está em OPSPILOT_CORS_ORIGINS.`,
    action: "open_settings",
  };
}

function fromStatus(status: number, body: unknown): UiError {
  const parsed = ApiErrorSchema.safeParse(body);
  const code = parsed.success ? parsed.data.error : undefined;
  const requestId = parsed.success ? parsed.data.requestId : undefined;
  const withId = (error: Omit<UiError, "requestId" | "status">): UiError =>
    requestId === undefined ? { ...error, status } : { ...error, requestId, status };

  if (status === 400) {
    return withId({ title: "Não deu para enviar essa mensagem", detail: "A API recusou o pedido. Revise o texto e tente de novo.", action: "retry" });
  }
  if (status === 404 && code === "conversation_not_found") {
    return withId({ title: "Essa conversa não existe mais", detail: "Comece uma nova conversa para continuar.", action: "new_conversation" });
  }
  if (status === 404 && code === "approval_not_found") {
    return withId({ title: "Ação não encontrada", detail: "Essa ação não existe mais na API. Nada foi executado.", action: "none" });
  }
  if (status === 409) {
    return withId({ title: "Ação já decidida", detail: "Outra decisão já foi registrada para essa ação.", action: "none" });
  }
  if (status === 410) {
    return withId({ title: "Ação expirada", detail: "O prazo para decidir passou. Nada foi executado; peça a ação de novo se ainda fizer sentido.", action: "none" });
  }
  if (status === 422) {
    return withId({ title: "Estratégia inválida", detail: "A estratégia pedida não existe na API.", action: "none" });
  }
  if (status === 504) {
    return withId(TIMEOUT);
  }
  return withId({ title: "O copiloto falhou ao responder", detail: "Algo deu errado na API. Tente de novo em instantes.", action: "retry" });
}

function fromException(exception: unknown, context: ErrorContext): UiError {
  if (exception instanceof Error && (exception.name === "AbortError" || exception.name === "TimeoutError")) {
    return { ...TIMEOUT };
  }
  if (exception instanceof Error && exception.name === "ZodError") {
    return { title: "A API respondeu num formato inesperado", detail: "A versão da API pode ser diferente da esperada pela war room. Tente de novo.", action: "retry" };
  }
  // `fetch` rejeita com TypeError em falha de rede e em bloqueio de CORS — o navegador não diferencia.
  return networkError(context);
}

/** Traduz qualquer falha de chamada à API numa mensagem para a tela. Pura. */
export function toUiError(input: ErrorInput, context: ErrorContext = {}): UiError {
  return "exception" in input ? fromException(input.exception, context) : fromStatus(input.status, input.body);
}

/** Motivo de um cartão que não aceita mais decisão (409/410/404), ou `undefined` se a falha é recuperável. */
export function unavailableReason(error: UiError): "Já decidida" | "Expirou" | "Não encontrada" | undefined {
  switch (error.status) {
    case 409:
      return "Já decidida";
    case 410:
      return "Expirou";
    case 404:
      return "Não encontrada";
    default:
      return undefined;
  }
}
