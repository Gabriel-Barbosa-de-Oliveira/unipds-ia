import {
  ChatAwaitingApprovalSchema,
  ChatOkSchema,
  DecisionOkSchema,
  type ChatAwaitingApproval,
  type ChatOk,
  type DecisionOk,
} from "../lib/api-schemas.ts";
import { toUiError, type UiError } from "../lib/errors.ts";

/** Pouco acima do teto de 180s da API, para o 504 dela chegar antes do abort local (research.md item 12). */
const CHAT_TIMEOUT_MS = 190_000;
const DECISION_TIMEOUT_MS = 30_000;
const PING_TIMEOUT_MS = 5_000;

export interface ChatRequest {
  message: string;
  conversationId?: string;
}

export type ChatResult =
  | { kind: "ok"; data: ChatOk }
  | { kind: "awaiting"; data: ChatAwaitingApproval }
  | { kind: "error"; error: UiError };

export type DecisionResult = { kind: "ok"; data: DecisionOk } | { kind: "error"; error: UiError };

const errorContext = () => ({ origin: typeof location === "undefined" ? undefined : location.origin });

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return undefined;
  }
}

/** Corpo de erro com o `requestId` do cabeçalho quando o corpo não trouxer (014). */
function errorBody(response: Response, body: unknown): unknown {
  const headerId = response.headers.get("X-Request-Id");
  if (headerId && body && typeof body === "object" && !("requestId" in body)) {
    return { ...body, requestId: headerId };
  }
  return body ?? (headerId ? { error: "unknown", requestId: headerId } : undefined);
}

function timeoutSignal(ms: number, external?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(ms);
  return external ? AbortSignal.any([timeout, external]) : timeout;
}

/** `POST /chat`. Nunca lança: toda falha volta como `UiError`. */
export async function postChat(apiUrl: string, request: ChatRequest, signal?: AbortSignal): Promise<ChatResult> {
  try {
    const response = await fetch(`${apiUrl}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: timeoutSignal(CHAT_TIMEOUT_MS, signal),
    });
    const body = await readJson(response);

    if (response.status === 200) {
      return { kind: "ok", data: ChatOkSchema.parse(body) };
    }
    if (response.status === 202) {
      return { kind: "awaiting", data: ChatAwaitingApprovalSchema.parse(body) };
    }
    return { kind: "error", error: toUiError({ status: response.status, body: errorBody(response, body) }) };
  } catch (exception) {
    return { kind: "error", error: toUiError({ exception }, errorContext()) };
  }
}

/** `POST /approvals/:id`. Nunca lança. */
export async function postDecision(
  apiUrl: string,
  approvalId: string,
  decision: "approve" | "deny",
): Promise<DecisionResult> {
  try {
    const response = await fetch(`${apiUrl}/approvals/${encodeURIComponent(approvalId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision }),
      signal: timeoutSignal(DECISION_TIMEOUT_MS),
    });
    const body = await readJson(response);

    if (response.status === 200) {
      return { kind: "ok", data: DecisionOkSchema.parse(body) };
    }
    return { kind: "error", error: toUiError({ status: response.status, body: errorBody(response, body) }) };
  } catch (exception) {
    return { kind: "error", error: toUiError({ exception }, errorContext()) };
  }
}

/** Teste de conexão da engrenagem: `GET /stats?since=1h` respondeu 200 (research.md item 11). Nunca lança. */
export async function ping(apiUrl: string): Promise<boolean> {
  try {
    const response = await fetch(`${apiUrl}/stats?since=1h`, { signal: timeoutSignal(PING_TIMEOUT_MS) });
    return response.status === 200;
  } catch {
    return false;
  }
}
