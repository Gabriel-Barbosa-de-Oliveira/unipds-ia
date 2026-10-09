import { z } from "zod";

export const DEFAULT_API_URL = "http://localhost:3000";

export const ThemePreferenceSchema = z.enum(["system", "light", "dark"]);
export type ThemePreference = z.infer<typeof ThemePreferenceSchema>;

export interface Settings {
  apiUrl: string;
  theme: ThemePreference;
}

export const DEFAULT_SETTINGS: Settings = { apiUrl: DEFAULT_API_URL, theme: "system" };

export type UrlResult = { ok: true; value: string } | { ok: false; error: string };

/** Valida e normaliza o endereço da API: URL absoluta http(s), sem espaços nem barra final. Pura. */
export function normalizeApiUrl(raw: string): UrlResult {
  const trimmed = raw.trim();
  if (trimmed.length === 0) {
    return { ok: false, error: "Informe o endereço da API." };
  }

  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    return { ok: false, error: "Use um endereço completo, por exemplo http://localhost:3000." };
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return { ok: false, error: "O endereço precisa começar com http:// ou https://." };
  }

  const path = url.pathname.replace(/\/+$/, "");
  return { ok: true, value: `${url.origin}${path}` };
}

/** Reconstrói as configurações a partir do que estava guardado; qualquer valor inválido cai no padrão. */
export function parseStoredSettings(raw: { apiUrl?: unknown; theme?: unknown }): Settings {
  const url = typeof raw.apiUrl === "string" ? normalizeApiUrl(raw.apiUrl) : undefined;
  const theme = ThemePreferenceSchema.safeParse(raw.theme);
  return {
    apiUrl: url?.ok ? url.value : DEFAULT_SETTINGS.apiUrl,
    theme: theme.success ? theme.data : DEFAULT_SETTINGS.theme,
  };
}

export function resolveTheme(preference: ThemePreference, prefersDark: boolean): "light" | "dark" {
  if (preference === "system") {
    return prefersDark ? "dark" : "light";
  }
  return preference;
}

export type ConnectionStatus = "idle" | "testing" | "connected" | "unreachable";

/** Estado do painel de configurações (US4). */
export interface SettingsFormState {
  draft: string;
  fieldError: string | null;
  saved: Settings;
  connection: ConnectionStatus;
}

export type SettingsFormAction =
  | { type: "edit"; value: string }
  | { type: "submit" }
  | { type: "restoreDefault" }
  | { type: "pingResult"; ok: boolean; apiUrl: string }
  | { type: "setTheme"; theme: ThemePreference }
  | { type: "open"; saved: Settings };

export function initialSettingsForm(saved: Settings): SettingsFormState {
  return { draft: saved.apiUrl, fieldError: null, saved, connection: "idle" };
}

/**
 * Reducer puro do painel. `submit` inválido mantém o endereço salvo (FR-021); válido salva e
 * passa a `testing` — o salvamento não depende do teste de conexão (FR-023).
 */
export function settingsFormReducer(state: SettingsFormState, action: SettingsFormAction): SettingsFormState {
  switch (action.type) {
    case "open":
      return initialSettingsForm(action.saved);
    case "edit":
      return { ...state, draft: action.value, fieldError: null };
    case "restoreDefault":
      return { ...state, draft: DEFAULT_API_URL, fieldError: null };
    case "submit": {
      const result = normalizeApiUrl(state.draft);
      if (!result.ok) {
        return { ...state, fieldError: result.error };
      }
      return {
        ...state,
        draft: result.value,
        fieldError: null,
        saved: { ...state.saved, apiUrl: result.value },
        connection: "testing",
      };
    }
    case "pingResult":
      // Resultado de um teste antigo (endereço já trocado) é ignorado.
      if (action.apiUrl !== state.saved.apiUrl) {
        return state;
      }
      return { ...state, connection: action.ok ? "connected" : "unreachable" };
    case "setTheme":
      return { ...state, saved: { ...state.saved, theme: action.theme } };
  }
}
