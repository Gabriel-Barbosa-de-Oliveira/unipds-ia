# Contract: `src/context/context-builder.ts`

Módulo puro (sem IO, sem `process.env`). Tipos detalhados em [data-model.md](../data-model.md).

## Exports

```ts
export interface ContextBudget { readonly summary: number; readonly window: number; readonly memories: number }
export const DEFAULT_CONTEXT_BUDGET: ContextBudget; // { summary: 200, window: 1200, memories: 300 }

export function loadContextBudget(env: Readonly<Record<string, string | undefined>>): ContextBudget;

export function trimSummary(summary: string, budget: number): string;
export function trimWindow(window: readonly ConversationMessage[], budget: number): ConversationMessage[];
export function trimMemories(memories: readonly ScoredMemory[], budget: number): ScoredMemory[];

export function buildContext(input: ContextInput, budget?: ContextBudget): BuiltContext; // budget default = DEFAULT_CONTEXT_BUDGET
```

## Comportamento

| Função | Regra |
|---|---|
| `loadContextBudget` | Lê `CONTEXT_BUDGET_SUMMARY`/`_WINDOW`/`_MEMORIES`; cada uma validada com zod (string não vazia → inteiro ≥ 0); inválida/ausente → padrão daquele campo. Nunca lança. |
| `trimSummary` | `estimateTokens(s) ≤ budget` → `s`; senão `s.slice(0, budget * 4)` (mantém o início). |
| `trimWindow` | Remove do início (mais antigas) até `estimateTokens(contents.join("\n")) ≤ budget`; preserva ordem; pode retornar `[]`. |
| `trimMemories` | Ordena por `score` desc (estável); remove do fim (menor score) até `estimateTokens(facts.join("\n")) ≤ budget`. |
| `buildContext` | Aplica os três cortes independentemente (sem empréstimo entre seções), mede o breakdown pós-corte, conta removidos e compõe `prompt` (abaixo). `system`/`message` nunca alterados. |

## Formato de `prompt`

Blocos unidos por `"\n\n"`, omitindo os ausentes, nesta ordem:

1. `system` (verbatim)
2. `"Resumo da conversa até aqui:\n" + summary` (pós-corte; omitido se vazio)
3. `composeWithFacts(memories.map(m => m.fact), composePrompt(window, message))`

Sem `system`/`summary`, `prompt` é exatamente o bloco 3 — idêntico ao `/chat` antes desta feature.

## Teste-âncora ("tetos baixos cortam na ordem certa")

Com budget `{ summary: 2, window: ≈2 mensagens, memories: ≈1 fato }` e entrada de 4 mensagens + 3 memórias com scores distintos (fornecidos fora de ordem), o resultado deve:
- manter só as **2 mensagens mais recentes**, em ordem cronológica (`trimmed.historyMessages = 2`);
- manter só a **memória de maior score** (`trimmed.recalledFacts = 2`);
- truncar o resumo para 8 caracteres;
- conter `system` e `message` íntegros em `prompt`, mesmo que maiores que todos os tetos.
