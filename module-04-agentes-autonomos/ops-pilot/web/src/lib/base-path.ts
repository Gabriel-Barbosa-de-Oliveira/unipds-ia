/**
 * Caminho base padrão da war room (015, FR-025) — usado no dev local. A publicação no Pages usa
 * `OPSPILOT_WEB_BASE` (016), ex.: `/unipds-ia/opspilot/`.
 */
export const BASE_PATH = "/opspilot/";

/**
 * Valida e normaliza `OPSPILOT_WEB_BASE` para o `base` do Vite (spec 016, research.md item 6).
 * Pura. Vazio cai no padrão; o que não for um caminho absoluto simples faz o build falhar.
 */
export function resolveBasePath(raw?: string): string {
  const trimmed = (raw ?? "").trim();
  if (trimmed.length === 0) {
    return BASE_PATH;
  }

  const invalid = /^[a-z][a-z0-9+.-]*:/i.test(trimmed) || /\s|[?#\\]/.test(trimmed) || trimmed.split("/").includes("..");
  if (invalid) {
    throw new Error(
      `OPSPILOT_WEB_BASE inválido: "${trimmed}". Use só um caminho, no formato /repo/opspilot/ (sem esquema, espaços, ?, # nem ..).`,
    );
  }

  const segments = trimmed.split("/").filter((segment) => segment.length > 0);
  return segments.length === 0 ? "/" : `/${segments.join("/")}/`;
}

/** `src`/`href` locais de um HTML (ignora URLs absolutas e âncoras). Pura. */
export function localAssetPaths(html: string): string[] {
  return [...html.matchAll(/\s(?:src|href)="([^"]+)"/g)]
    .map((match) => match[1] ?? "")
    .filter((path) => path.length > 0 && !/^(?:[a-z]+:)?\/\//i.test(path) && !path.startsWith("#") && !path.startsWith("data:"));
}
