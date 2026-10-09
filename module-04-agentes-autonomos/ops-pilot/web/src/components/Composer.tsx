import { useId, useState, type FormEvent, type KeyboardEvent } from "react";

interface ComposerProps {
  /** Devolve `true` quando o envio foi aceito; só então o texto é limpo (FR-004). */
  onSend: (text: string) => boolean;
  disabled: boolean;
  disabledReason?: string;
}

export function Composer({ onSend, disabled, disabledReason }: ComposerProps) {
  const [text, setText] = useState("");
  const inputId = useId();
  const noteId = useId();
  const canSend = !disabled && text.trim().length > 0;

  function submit(event?: FormEvent) {
    event?.preventDefault();
    if (canSend && onSend(text)) {
      setText("");
    }
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <form className="composer" onSubmit={submit}>
      <div className="composer-inner">
        <label className="label" htmlFor={inputId}>
          Mensagem
        </label>
        <div className="composer-row">
          <textarea
            id={inputId}
            className="textarea"
            rows={2}
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Pergunte sobre alertas, incidentes ou runbooks"
            aria-describedby={noteId}
          />
          <button type="submit" className="btn btn-primary" disabled={!canSend}>
            Enviar
          </button>
        </div>
        <p id={noteId} className="hint">
          {disabled && disabledReason ? disabledReason : "Enter envia · Shift+Enter quebra a linha"}
        </p>
      </div>
    </form>
  );
}
