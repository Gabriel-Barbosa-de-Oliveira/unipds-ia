import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  DEFAULT_API_URL,
  initialSettingsForm,
  normalizeApiUrl,
  parseStoredSettings,
  resolveTheme,
  settingsFormReducer,
} from "./settings.ts";

describe("normalizeApiUrl", () => {
  test("remove a barra final", () => {
    assert.deepEqual(normalizeApiUrl("http://localhost:3000/"), { ok: true, value: "http://localhost:3000" });
  });

  test("tira espaços e preserva o caminho base", () => {
    assert.deepEqual(normalizeApiUrl(" https://api.x.com/base/ "), { ok: true, value: "https://api.x.com/base" });
  });

  test("rejeita endereço sem esquema, com esquema errado ou vazio, com mensagem humana", () => {
    for (const raw of ["localhost:3000", "ftp://x.com", "", "   ", "não é url"]) {
      const result = normalizeApiUrl(raw);
      assert.equal(result.ok, false, raw);
      if (!result.ok) {
        assert.ok(result.error.length > 0);
      }
    }
  });
});

describe("parseStoredSettings", () => {
  test("valores inválidos caem no padrão", () => {
    assert.deepEqual(parseStoredSettings({ apiUrl: "lixo", theme: "roxo" }), { apiUrl: DEFAULT_API_URL, theme: "system" });
    assert.deepEqual(parseStoredSettings({}), { apiUrl: DEFAULT_API_URL, theme: "system" });
  });

  test("valores válidos são normalizados", () => {
    assert.deepEqual(parseStoredSettings({ apiUrl: "https://ops.x.com/", theme: "dark" }), {
      apiUrl: "https://ops.x.com",
      theme: "dark",
    });
  });
});

describe("resolveTheme", () => {
  test("system segue o sistema; escolha manual vence", () => {
    assert.equal(resolveTheme("system", true), "dark");
    assert.equal(resolveTheme("system", false), "light");
    assert.equal(resolveTheme("light", true), "light");
    assert.equal(resolveTheme("dark", false), "dark");
  });
});

describe("settingsFormReducer", () => {
  const saved = { apiUrl: "http://localhost:3000", theme: "system" as const };

  test("submit inválido mostra erro no campo e mantém o endereço salvo (FR-021)", () => {
    let state = initialSettingsForm(saved);
    state = settingsFormReducer(state, { type: "edit", value: "localhost:3999" });
    state = settingsFormReducer(state, { type: "submit" });
    assert.ok(state.fieldError);
    assert.equal(state.saved.apiUrl, "http://localhost:3000");
    assert.equal(state.connection, "idle");
  });

  test("submit válido salva normalizado e passa a testar a conexão", () => {
    let state = initialSettingsForm(saved);
    state = settingsFormReducer(state, { type: "edit", value: "http://localhost:3999/" });
    state = settingsFormReducer(state, { type: "submit" });
    assert.equal(state.fieldError, null);
    assert.equal(state.saved.apiUrl, "http://localhost:3999");
    assert.equal(state.connection, "testing");
  });

  test("pingResult define conectado/inacessível só para o endereço atual", () => {
    let state = settingsFormReducer(initialSettingsForm(saved), { type: "submit" });
    assert.equal(settingsFormReducer(state, { type: "pingResult", ok: true, apiUrl: saved.apiUrl }).connection, "connected");
    assert.equal(settingsFormReducer(state, { type: "pingResult", ok: false, apiUrl: saved.apiUrl }).connection, "unreachable");
    state = settingsFormReducer(state, { type: "pingResult", ok: true, apiUrl: "http://outro" });
    assert.equal(state.connection, "testing");
  });

  test("editar limpa o erro; restaurar padrão preenche o campo", () => {
    let state = settingsFormReducer(initialSettingsForm(saved), { type: "edit", value: "x" });
    state = settingsFormReducer(state, { type: "submit" });
    state = settingsFormReducer(state, { type: "edit", value: "xy" });
    assert.equal(state.fieldError, null);
    state = settingsFormReducer(state, { type: "restoreDefault" });
    assert.equal(state.draft, DEFAULT_API_URL);
  });

  test("setTheme atualiza o tema salvo", () => {
    const state = settingsFormReducer(initialSettingsForm(saved), { type: "setTheme", theme: "dark" });
    assert.equal(state.saved.theme, "dark");
  });
});
