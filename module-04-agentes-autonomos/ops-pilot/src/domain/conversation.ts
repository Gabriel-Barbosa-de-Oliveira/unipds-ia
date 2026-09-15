export type ConversationRole = "user" | "assistant";

export interface ConversationMessage {
  readonly role: ConversationRole;
  readonly content: string;
}

const ROLE_LABELS: Record<ConversationRole, string> = {
  user: "Pessoa de plantão",
  assistant: "Copiloto",
};

/**
 * Compõe o histórico de uma conversa com a nova mensagem em uma única string — é isso que vira
 * o `input` já hoje aceito por `ReasoningStrategy.run(input, options)` (research.md item 1),
 * então nenhuma estratégia precisa saber que existe uma conversa por trás. Sem histórico,
 * retorna `input` inalterado. Pura: mesma entrada sempre produz a mesma string.
 */
export function composePrompt(history: readonly ConversationMessage[], input: string): string {
  if (history.length === 0) {
    return input;
  }

  const historyText = history
    .map((message) => `${ROLE_LABELS[message.role]}: ${message.content}`)
    .join("\n");

  return [
    "Histórico da conversa até aqui:",
    historyText,
    "",
    `Mensagem atual da pessoa de plantão: ${input}`,
  ].join("\n");
}
