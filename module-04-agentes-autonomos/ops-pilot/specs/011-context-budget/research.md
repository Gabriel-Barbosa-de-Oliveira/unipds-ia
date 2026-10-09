# Research: Orçamento de Contexto por Seção

## 1. Como "montar o prompt de TODAS as estratégias" sem mexer em cada estratégia

- **Decision**: o builder produz a `string` passada a `ReasoningStrategy.run(input)`. Estratégias não mudam.
- **Rationale**: react (`messages: [{ role: "user", content: input }]`), plan-and-execute (planner/replanner usam `state.input`) e `withReflection` (repassa `input` à estratégia base e ao crítico) já recebem o contexto composto exclusivamente por esse `input` — é o mesmo mecanismo de `006`/`007`. Montar o `input` num ponto único cobre todas as estratégias, atuais e futuras, com zero mudança nelas.
- **Alternatives considered**: (a) builder retornar `BaseMessage[]` com system separado e alterar `ReasoningStrategy.run` — quebra o contrato de `001`, mexe em 3 estratégias + bench/arena, sem ganho para esta feature; (b) cortar dentro de cada estratégia — duplicação e risco de divergência.

## 2. Seção "system"

- **Decision**: `buildContext` aceita `system?: string`, incluído integralmente no topo quando presente; o `/chat` hoje não fornece nenhum (as estratégias não têm system prompt próprio; o `CRITIC_PROMPT` da reflection é interno ao crítico e fica fora do builder).
- **Rationale**: atende "system intocável" sem inventar um system prompt novo (o que mudaria o comportamento do raciocínio). Com `system` ausente, FR-011 (saída idêntica) se mantém.
- **Alternatives considered**: criar um system prompt de plantão agora — fora do escopo da spec.

## 3. Seção "resumo"

- **Decision**: `summary?: string`, opcional; o `/chat` não fornece (não existe gerador de resumo). Corte: `text.slice(0, budget * 4)` — mantém o início.
- **Rationale**: `estimateTokens(s) = ceil(len/4) ≤ budget ⟺ len ≤ 4·budget`, então o corte é exato na unidade usada. Manter o início segue a Assumption da spec.
- **Alternatives considered**: cortar por palavra/frase — mais "bonito", mas não exato e não pedido.

## 4. Unidade de medida e convenção da soma

- **Decision**: uma seção de itens é medida como `estimateTokens(items.join("\n"))` — exatamente a convenção de `buildContextBreakdown` (`009`). O teto é verificado com essa mesma medida.
- **Rationale**: garante que o `contextBreakdown` retornado nunca mostra uma seção acima do seu teto (SC-001 verificável pela própria resposta). Somar `estimateTokens` por item superestimaria por arredondamento e divergiria do breakdown.
- **Alternatives considered**: soma por item — simples, mas incoerente com as métricas da `009`.

## 5. Algoritmo de corte da janela

- **Decision**: enquanto `measure(window) > budget`, remove `window[0]` (a mais antiga; `lastMessages` já retorna em ordem cronológica). Resultado preserva a ordem. Se a mais recente sozinha excede, a janela fica vazia.
- **Rationale**: literal ao pedido ("corta as mais antigas") e à FR-005 (itens inteiros). Não "pula" uma mensagem grande para manter uma mais antiga — isso quebraria a continuidade da conversa.
- **Alternatives considered**: greedy do mais novo pulando itens grandes — deixa buracos no meio do diálogo.

## 6. Algoritmo de corte das memórias

- **Decision**: ordena por `score` desc com sort estável (empates mantêm a ordem de entrada) e, enquanto exceder, remove o último (menor score). A seção final é emitida em ordem de score desc (mesma ordem que `recall` já devolve hoje).
- **Rationale**: literal ("corta menor score"), determinístico (FR-009, edge case de empate). `Array.prototype.sort` é estável desde ES2019.
- **Alternatives considered**: knapsack (maximizar score total no teto) — complexidade sem pedido.

## 7. Composição final (FR-011)

- **Decision**: `prompt = [system?, summaryBlock?, composeWithFacts(keptFacts, composePrompt(keptWindow, message))]` unidos por `"\n\n"`, onde `summaryBlock = "Resumo da conversa até aqui:\n" + summary`. Sem system e sem resumo, o resultado é exatamente `composeWithFacts(facts, composePrompt(history, message))` — o que `server.ts` faz hoje.
- **Rationale**: reaproveita as funções puras existentes; teste de regressão compara diretamente as duas strings.

## 8. Variáveis de ambiente

- **Decision**: `CONTEXT_BUDGET_SUMMARY`, `CONTEXT_BUDGET_WINDOW`, `CONTEXT_BUDGET_MEMORIES`. `loadContextBudget(env: Record<string, string | undefined>)` valida cada uma com zod (`string` não vazia → `z.coerce.number().int().min(0)`); qualquer falha → padrão da seção (200/1200/300). Chamado uma vez em `createApp` com `process.env`; `CreateAppOptions.contextBudget` sobrescreve (testes).
- **Rationale**: env é entrada externa (Princípio II); receber `env` por parâmetro mantém a função pura (Princípio IV). Atenção: `z.coerce.number()` transforma `""` em `0` — por isso string vazia é tratada como ausente antes da coerção. `0` explícito é válido e desliga a seção (edge case da spec).
- **Alternatives considered**: lançar erro na inicialização com env inválida — a spec pede fallback silencioso.

## 9. Métricas

- **Decision**: `contextBreakdown` passa a ter `system` e `summary` (0 quando ausentes) além de `currentMessage`/`conversationHistory`/`recalledFacts`, todos **pós-corte**; `total` continua sendo a soma. Novo `metrics.contextTrimmed = { historyMessages, recalledFacts }` (quantidades removidas). `metrics.historyMessages` passa a contar as mensagens efetivamente enviadas (pós-corte).
- **Rationale**: FR-010; `historyMessages` sempre significou "mensagens de histórico consideradas no prompt" (`006`) — com tetos padrão e `HISTORY_LIMIT = 12` o valor só muda quando há corte real. Mudança aditiva; os asserts `deepEqual` existentes em `server.test.ts` precisam ganhar as chaves novas.
- **Alternatives considered**: manter `contextBreakdown` com as 3 chaves antigas — esconderia system/resumo do total.

## 10. arena/bench

- **Decision**: passam `buildContext({ message: input }).prompt` (com o budget padrão) — saída idêntica a `input`.
- **Rationale**: honra "TODAS as estratégias montadas pelo builder" sem alterar resultado algum de benchmark.
