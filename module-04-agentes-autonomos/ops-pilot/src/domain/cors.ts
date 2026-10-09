/** Origem do dev server da war room — a única liberada quando nada é configurado (spec 015, FR-028). */
export const DEFAULT_CORS_ORIGINS: readonly string[] = ["http://localhost:5173"];

/** Lê `OPSPILOT_CORS_ORIGINS` (lista separada por vírgula). Pura; vazio cai no padrão. */
export function parseAllowedOrigins(raw: string | undefined): string[] {
  const origins = (raw ?? "")
    .split(",")
    .map((origin) => origin.trim())
    .filter((origin) => origin.length > 0);
  return origins.length > 0 ? origins : [...DEFAULT_CORS_ORIGINS];
}

/**
 * Cabeçalhos CORS para uma requisição vinda de `origin`. Comparação exata, sem curinga e sem
 * credenciais (research.md item 6): origem fora da lista não recebe nada e o navegador bloqueia.
 */
export function corsHeadersFor(
  origin: string | undefined,
  allowlist: readonly string[],
  opts: { preflight: boolean },
): Record<string, string> {
  if (origin === undefined || !allowlist.includes(origin)) {
    return {};
  }

  const headers: Record<string, string> = {
    "Access-Control-Allow-Origin": origin,
    Vary: "Origin",
    "Access-Control-Expose-Headers": "X-Request-Id",
  };
  if (opts.preflight) {
    headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
    headers["Access-Control-Allow-Headers"] = "Content-Type";
    headers["Access-Control-Max-Age"] = "600";
  }
  return headers;
}
