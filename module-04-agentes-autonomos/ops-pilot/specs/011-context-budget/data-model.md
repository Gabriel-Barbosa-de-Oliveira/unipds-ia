# Data Model: Orçamento de Contexto por Seção

Todas as estruturas são transientes (por requisição), imutáveis (`readonly`) e sem persistência.

## ContextBudget

| Campo | Tipo | Padrão | Env | Regra |
|---|---|---|---|---|
| `summary` | number | 200 | `CONTEXT_BUDGET_SUMMARY` | inteiro ≥ 0; tokens estimados |
| `window` | number | 1200 | `CONTEXT_BUDGET_WINDOW` | inteiro ≥ 0 |
| `memories` | number | 300 | `CONTEXT_BUDGET_MEMORIES` | inteiro ≥ 0 |

- Valor ausente, vazio, não numérico, fracionário ou negativo → padrão do campo (por campo, independente).
- `0` → seção sempre vazia.
- System e mensagem atual **não** têm teto (intocáveis).

## ContextInput (entrada de `buildContext`)

| Campo | Tipo | Obrigatório | Origem no `/chat` |
|---|---|---|---|
| `message` | string | sim | `ChatRequestSchema.message` |
| `system` | string | não | (não fornecido hoje) |
| `summary` | string | não | (não fornecido hoje) |
| `window` | `readonly ConversationMessage[]` | não (default `[]`) | `conversationStore.lastMessages(id, 12)` — ordem cronológica |
| `memories` | `readonly ScoredMemory[]` | não (default `[]`) | `memoryStore.recall(...)` |

### ScoredMemory

| Campo | Tipo | Regra |
|---|---|---|
| `fact` | string | texto do fato |
| `score` | number | relevância; maior = mais relevante (mesmo `RecallMatch` de `007`) |

## BuiltContext (saída de `buildContext`)

| Campo | Tipo | Regra |
|---|---|---|
| `prompt` | string | texto final passado a `ReasoningStrategy.run` |
| `window` | `readonly ConversationMessage[]` | mensagens mantidas, ordem cronológica |
| `memories` | `readonly ScoredMemory[]` | memórias mantidas, score desc |
| `summary` | string | resumo pós-corte (`""` se ausente) |
| `breakdown` | `ContextBreakdown` | tamanhos pós-corte (abaixo) |
| `trimmed` | `{ historyMessages: number; recalledFacts: number }` | quantidades removidas |

### Invariantes

- `prompt` contém `system` e `message` exatamente como recebidos.
- `estimateTokens(summary) ≤ budget.summary`; `estimateTokens(window.map(c).join("\n")) ≤ budget.window`; `estimateTokens(memories.map(f).join("\n")) ≤ budget.memories`.
- `window` é sufixo da janela de entrada (só as mais antigas saem).
- `memories` é prefixo da entrada ordenada por score desc (estável) — só as de menor score saem.
- `trimmed.historyMessages = input.window.length − window.length`; idem para memórias.
- Nenhum corte ⇒ `prompt` idêntico a `composeWithFacts(facts, composePrompt(window, message))` quando sem system/resumo.

## ContextBreakdown (estendido de `009`)

| Campo | Desde | Regra |
|---|---|---|
| `system` | **010** | `estimateTokens(system)`; 0 se ausente |
| `summary` | **010** | `estimateTokens(summary pós-corte)`; 0 se ausente |
| `currentMessage` | 009 | inalterado |
| `conversationHistory` | 009 | agora sobre a janela **pós-corte** |
| `recalledFacts` | 009 | agora sobre as memórias **pós-corte** |
| `total` | 009 | soma de todas as partes acima |
