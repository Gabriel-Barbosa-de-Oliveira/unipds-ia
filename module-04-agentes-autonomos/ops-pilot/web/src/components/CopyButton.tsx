import { useState } from "react";

import { Icon } from "./Icon.tsx";

/** Copia um valor (ex.: id da requisição). Falha de clipboard é silenciosa: o valor continua visível. */
export function CopyButton({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }

  return (
    <button type="button" className="btn btn-ghost" onClick={copy} aria-label={label}>
      <Icon name={copied ? "check" : "copy"} />
      <span aria-live="polite">{copied ? "Copiado" : "Copiar"}</span>
    </button>
  );
}

export function RequestId({ id }: { id: string }) {
  return (
    <p className="request-id">
      <span>ID da requisição:</span>
      <code>{id}</code>
      <CopyButton value={id} label="Copiar ID da requisição" />
    </p>
  );
}
