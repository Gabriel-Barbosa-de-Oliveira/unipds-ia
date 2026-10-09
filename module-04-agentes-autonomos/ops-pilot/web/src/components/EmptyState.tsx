export const EXAMPLES = [
  "Quais alertas estão disparando?",
  "Quais incidentes estão abertos?",
  "Qual o runbook do checkout-api?",
] as const;

/** Conversa vazia: explica o que dá para perguntar e oferece exemplos clicáveis (US1, cenário 3). */
export function EmptyState({ onPick, disabled }: { onPick: (text: string) => void; disabled: boolean }) {
  return (
    <section className="empty" aria-labelledby="empty-title">
      <h2 id="empty-title">Pronto para o plantão</h2>
      <p>
        Pergunte sobre alertas, incidentes e runbooks. O copiloto consulta a produção e pede sua aprovação antes de
        abrir ou resolver qualquer incidente.
      </p>
      <ul className="examples" aria-label="Exemplos de perguntas">
        {EXAMPLES.map((example) => (
          <li key={example}>
            <button type="button" className="btn btn-secondary" onClick={() => onPick(example)} disabled={disabled}>
              {example}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
