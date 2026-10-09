import { useEffect, useId, useReducer, type FormEvent } from "react";

import { ping } from "../api/client.ts";
import {
  initialSettingsForm,
  settingsFormReducer,
  type Settings,
  type ThemePreference,
} from "../lib/settings.ts";
import { Dialog } from "./Dialog.tsx";
import { Icon } from "./Icon.tsx";

interface SettingsPanelProps {
  settings: Settings;
  onChange: (settings: Settings) => void;
  onClose: () => void;
}

const THEMES: { value: ThemePreference; label: string }[] = [
  { value: "system", label: "Seguir o sistema" },
  { value: "light", label: "Claro" },
  { value: "dark", label: "Escuro" },
];

/** Engrenagem: endereço da API (validado, testado, salvo) e tema (US4). */
export function SettingsPanel({ settings, onChange, onClose }: SettingsPanelProps) {
  const [state, dispatch] = useReducer(settingsFormReducer, settings, initialSettingsForm);
  const inputId = useId();
  const helpId = useId();
  const errorId = useId();

  useEffect(() => {
    if (state.connection !== "testing") {
      return;
    }
    const apiUrl = state.saved.apiUrl;
    let active = true;
    void ping(apiUrl).then((ok) => {
      if (active) {
        dispatch({ type: "pingResult", ok, apiUrl });
      }
    });
    return () => {
      active = false;
    };
  }, [state.connection, state.saved.apiUrl]);

  useEffect(() => {
    if (state.saved.apiUrl !== settings.apiUrl || state.saved.theme !== settings.theme) {
      onChange(state.saved);
    }
  }, [state.saved, settings, onChange]);

  function submit(event: FormEvent) {
    event.preventDefault();
    dispatch({ type: "submit" });
  }

  return (
    <Dialog title="Configurações" onClose={onClose}>
      <form className="field" onSubmit={submit} noValidate>
        <label className="label" htmlFor={inputId}>
          Endereço da API
        </label>
        <input
          id={inputId}
          className="input"
          type="url"
          inputMode="url"
          autoComplete="url"
          value={state.draft}
          onChange={(event) => dispatch({ type: "edit", value: event.target.value })}
          aria-invalid={state.fieldError ? "true" : "false"}
          aria-describedby={state.fieldError ? `${errorId} ${helpId}` : helpId}
        />
        {state.fieldError && (
          <p id={errorId} className="inline-error">
            {state.fieldError}
          </p>
        )}
        <p id={helpId} className="hint">
          Em uso: <code>{state.saved.apiUrl}</code>. Fica guardado neste navegador.
        </p>
        <div className="form-actions">
          <button type="submit" className="btn btn-primary">
            Salvar
          </button>
          <button type="button" className="btn btn-secondary" onClick={() => dispatch({ type: "restoreDefault" })}>
            Restaurar padrão
          </button>
          <button type="button" className="btn btn-ghost" onClick={onClose}>
            Cancelar
          </button>
        </div>
        <p className="connection" role="status">
          {state.connection === "testing" && (
            <>
              <Icon name="clock" /> Testando conexão…
            </>
          )}
          {state.connection === "connected" && (
            <span className="status-success connection">
              <Icon name="check" /> Conectado
            </span>
          )}
          {state.connection === "unreachable" && (
            <span className="status-danger connection">
              <Icon name="cross" /> Não foi possível conectar. O endereço foi salvo mesmo assim.
            </span>
          )}
        </p>
      </form>

      <fieldset className="fieldset">
        <legend className="label">Tema</legend>
        {THEMES.map((theme) => (
          <label key={theme.value} className="radio">
            <input
              type="radio"
              name="theme"
              value={theme.value}
              checked={state.saved.theme === theme.value}
              onChange={() => dispatch({ type: "setTheme", theme: theme.value })}
            />
            {theme.label}
          </label>
        ))}
      </fieldset>
    </Dialog>
  );
}
