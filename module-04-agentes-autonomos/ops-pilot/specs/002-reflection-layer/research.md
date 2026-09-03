# Phase 0 Research: Camada de Reflection

## 1. Onde `maxReflections` é configurado

**Decision**: `maxReflections` é um parâmetro de construção do decorador — `withReflection(strategy, opts?: { maxReflections?: number })` —, não um campo novo em `RunOptions`. Ao registrar as estratégias na arena, `reflect:react`/`reflect:plan-and-execute` já nascem com o `maxReflections` desejado (padrão 2 quando `opts` é omitido); `RunOptions.maxIterations` continua sendo repassado por `run(input, options)` a cada tentativa, controlando o limite interno da estratégia base como já ocorre hoje.

**Rationale**: A assinatura pedida explicitamente foi `withReflection(strategy, opts)` — `opts` no momento da decoração, não do `run()`. Isso também evita alterar o contrato comum `RunOptions`/`ReasoningStrategy` (que `react.ts` e `plan-and-execute.ts` já implementam) só para acomodar um conceito que só existe no wrapper; `maxIterations` (limite de passos de uma tentativa) e `maxReflections` (limite de tentativas) são dimensões independentes e não deveriam compartilhar o mesmo canal.

**Alternatives considered**:
- *Adicionar `maxReflections` a `RunOptions`*: forçaria toda estratégia (inclusive as que nunca serão decoradas) a conhecer um conceito que só faz sentido para quem as decora; rejeitado por vazar um detalhe do decorador para o contrato comum.
- *Configurar só via variável de ambiente*: menos explícito, dificulta comparar diferentes limites na mesma arena; rejeitado.

## 2. Crítico: modelo, prompt e validação da saída estruturada

**Decision**: O crítico reaproveita `createModel()` (mesma fábrica única, `temperature: 0`) e usa `withStructuredOutput` com um schema zod `CritiqueSchema = z.object({ approved: z.boolean(), feedback: z.string().min(1) })` — mesmo padrão já usado por `PlanSchema`/`ReplanSchema` em `plan-and-execute.ts`. O prompt do crítico é montado por uma função pura `buildCritiquePrompt(input, trace, answer)` que inclui a pergunta original, o trace formatado da tentativa (via `formatTrace`, já existente em `trace.ts`) e a resposta final produzida, pedindo para o crítico decidir se a resposta é sustentada pelas observações do trace.

**Rationale**: Reaproveitar `createModel()` mantém uma única fonte de verdade para credenciais/`temperature` (Principle VI); reaproveitar `withStructuredOutput` + zod é o mesmo mecanismo já validado pela feature 001 para forçar saída estruturada e tratável como fronteira de confiança (Principle II) — a saída do modelo não é tratada como decisão de fluxo até passar pelo schema.

**Alternatives considered**:
- *Parsear JSON manualmente da resposta de texto do crítico*: reintroduz o problema que `withStructuredOutput` já resolve (parsing frágil); rejeitado.
- *Um modelo/serviço de crítica separado (ex.: heurística sem LLM)*: contradiz "mesmo modelo" do pedido original; rejeitado.
- *Crítico avalia apenas a resposta final, sem o trace*: mais simples, mas não cumpre "avalia a resposta contra as observações do trace" (FR-002) — perderia a checagem de que a resposta é de fato sustentada pelo que foi observado.

## 3. Como o feedback chega na tentativa seguinte

**Decision**: Uma função pura `buildRetryInput(originalInput, previousAnswer, feedback)` monta uma nova string de entrada que inclui o pedido original, a tentativa anterior e o motivo da reprovação, e essa string (não a original) é o `input` passado para `strategy.run()` na tentativa seguinte. `RunOptions` (incluindo `maxIterations`) é repassado inalterado a cada tentativa.

**Rationale**: A interface `ReasoningStrategy.run(input: string, options?: RunOptions)` (contrato da feature 001) não tem um canal dedicado para "contexto adicional" — o único canal de entrada é a própria string `input`. Embutir o feedback no texto de entrada cumpre FR-003 ("incorporando o feedback... ao contexto dessa nova tentativa") sem exigir nenhuma mudança na interface comum nem nas estratégias já implementadas (FR-001: decorar sem alterar a estratégia envolvida).

**Alternatives considered**:
- *Estender `RunOptions` com um campo `priorFeedback`*: obrigaria `react.ts`/`plan-and-execute.ts` a saber interpretar um campo que hoje não existe, violando FR-001 (a estratégia envolvida não deveria precisar de nenhuma alteração); rejeitado.
- *Manter um histórico de mensagens fora da string `input` (ex.: um array de mensagens passado por fora do contrato)*: exigiria um canal novo na interface `ReasoningStrategy`, mesmo problema acima; rejeitado.

## 4. Ciclo de reflection como orquestração pura testável

**Decision**: A lógica do ciclo fica em `runReflectionLoop(runAttempt, critique, input, options, maxReflections)`, uma função assíncrona que recebe como parâmetros a chamada de tentativa (`(input, options) => Promise<RunResult>`) e a chamada de crítica (`(attempt: RunResult, input: string) => Promise<{ approved: boolean; feedback: string }>`) já resolvidas — ela mesma não instancia modelo nem chama `createModel()`. `withReflection(strategy, opts)` é a única função que conecta `strategy.run` e um crítico real (item 2) a essa função. O loop: roda a 1ª tentativa; enquanto o crítico reprovar e o número de regenerações já feitas for menor que `maxReflections`, gera uma nova tentativa com `buildRetryInput` e repete a crítica; para na primeira aprovação ou ao esgotar `maxReflections` (contando regenerações extras após a 1ª tentativa — ver spec FR-005, esclarecido com o usuário: até `maxReflections + 1` tentativas no total, padrão 3).

**Rationale**: Isolar a orquestração de decisão (quantas tentativas, quando parar) da IO real (chamar o modelo/a estratégia) é o mesmo padrão de separação já adotado pela constitution (Principle IV) e pela feature 001 (`ops-store.ts` puro vs. adaptadores com IO). Torna `runReflectionLoop` testável de ponta a ponta com fakes determinísticos (aprovação imediata, reprovação N vezes, `maxReflections = 0`) sem precisar de um modelo real nem de mocks pesados de rede (Principle V).

**Alternatives considered**:
- *Toda a lógica dentro de `withReflection`, sem separar a função pura*: mais direto de escrever, mas exigiria mockar `createModel()`/rede para testar qualquer variação do ciclo (aprovação/reprovação/limite); rejeitado por dificultar testes determinísticos.

## 5. Agregação de métricas e trace

**Decision**: `llmCalls` da execução com reflection = soma de `metrics.llmCalls` de cada tentativa (`RunResult` retornado por `strategy.run` a cada chamada) + 1 chamada por avaliação crítica realizada (cada crítica é sempre exatamente uma invocação estruturada ao modelo). `latencyMs` é medido de ponta a ponta pelo próprio wrapper (`startTimer()` no início de `withReflection.run()`, igual ao padrão já usado por `react.ts`/`plan-and-execute.ts`), não pela soma das latências individuais reportadas por cada tentativa — evita contar tempo de forma inconsistente entre estratégias. O trace final é a concatenação cronológica de: trace da tentativa 1 → evento `critique` da 1ª crítica → (se reprovado) trace da tentativa 2 → evento `critique` da 2ª crítica → ... — cada evento é reindexado (`at`) para manter a ordem crescente contínua ao longo de todo o histórico, do mesmo jeito que `messagesToTrace(messages, startAt)` já reindexa por segmento na feature 001.

**Rationale**: Medir latência no ponto de entrada/saída comum do wrapper segue a mesma decisão já tomada na feature 001 (research.md item 5) — uma única definição de "tempo total", não uma soma sujeita a overhead de cada chamada individual. Somar `llmCalls` de todas as tentativas mais o crítico cumpre FR-008 diretamente. Reindexar `at` ao concatenar preserva a garantia de trace cronologicamente ordenado exigida pelo contrato de `ReasoningStrategy` (feature 001, contracts/reasoning-strategy.md).

**Alternatives considered**:
- *Somar as latências reportadas por cada tentativa*: viés de dupla contagem/lacunas (não inclui o tempo da própria chamada ao crítico entre tentativas); rejeitado.
- *Não reindexar `at` entre segmentos de trace*: geraria índices duplicados/fora de ordem entre tentativas, quebrando a garantia de "trace ordenado cronologicamente"; rejeitado.

## 6. Exposição na arena

**Decision**: `src/arena.ts` estende `STRATEGY_NAMES` para incluir `"reflect:react"` e `"reflect:plan-and-execute"`, e o registro `STRATEGIES` passa a incluir `"reflect:react": withReflection(reactStrategy)` e `"reflect:plan-and-execute": withReflection(planAndExecuteStrategy)` (ambos com `maxReflections` padrão, já que a feature não pede uma flag de CLI dedicada para configurá-lo). As estratégias `react`/`plan-and-execute` continuam registradas e inalteradas — a adição é puramente aditiva.

**Rationale**: Cumpre FR-009/FR-010 (nomes distintos, versões base preservadas) e a User Story 2 (comparar base vs. reflection na mesma arena) sem exigir nenhuma flag nova — `--strategies reflect:react,react` já funciona com o parsing zod existente (`z.enum(STRATEGY_NAMES)`), já que zod aceita strings com `:` como valor de enum sem tratamento especial.

**Alternatives considered**:
- *Uma flag `--max-reflections` dedicada na CLI*: adicionaria superfície nova não pedida pela spec (que não lista uma User Story/FR para configurar `maxReflections` via arena); mantido fora de escopo — pode ser proposto depois se necessário.
- *Substituir as estratégias base por versões sempre refletidas*: contradiz a Assumption da spec ("as estratégias base permanecem disponíveis e inalteradas") e elimina a possibilidade de comparação da User Story 2; rejeitado.
