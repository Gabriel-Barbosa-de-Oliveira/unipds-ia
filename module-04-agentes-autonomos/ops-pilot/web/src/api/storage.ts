import { DEFAULT_SETTINGS, parseStoredSettings, type Settings } from "../lib/settings.ts";

const API_URL_KEY = "opspilot.apiUrl";
const THEME_KEY = "opspilot.theme";

/** Lê as configurações do navegador; armazenamento bloqueado ou vazio cai no padrão (research.md item 14). */
export function loadSettings(): Settings {
  try {
    return parseStoredSettings({
      apiUrl: localStorage.getItem(API_URL_KEY) ?? undefined,
      theme: localStorage.getItem(THEME_KEY) ?? undefined,
    });
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(settings: Settings): void {
  try {
    localStorage.setItem(API_URL_KEY, settings.apiUrl);
    localStorage.setItem(THEME_KEY, settings.theme);
  } catch {
    // Sem armazenamento (aba privada, bloqueio): vale só para esta sessão.
  }
}
