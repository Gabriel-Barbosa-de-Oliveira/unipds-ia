# Research: Refletor de Aprendizado

## 1. Onde o refletor é disparado

**Decision**: `src/http/server.ts`, dentro do handler de `POST /chat` — depois que `result` (a resposta da estratégia de raciocínio) já foi calculado, e só quando `userId` está presente na requisição. O disparo não é `await`ado: `void reflect(memoryStore, userId, parsed.data.message).catch(() => {})` roda em paralelo enquanto a resposta HTTP já é montada e enviada.

**Rationale**: a spec (FR-001) fala em analisar "a última mensagem da pessoa" depois de cada resposta — isso não depende do conteúdo de `answer`, só de `parsed.data.message` (já validado pelo `ChatRequestSchema`). Exigir `userId` é o mesmo precondicional já usado por `recall`/`remember_fact`/`forget_fact` (`007-semantic-memory`): sem `userId` não há em quem registrar o fato, então nada é analisado — zero custo extra (nenhuma chamada de modelo) no caminho sem `userId`, mesmo espírito de `007-semantic-memory` research.md item 7 (comportamento idêntico ao de antes quando a feature não se aplica).

**Alternatives considered**: rodar a análise síncrona, antes de responder. Rejeitada: violaria FR-005/SC-003 (a resposta não pode esperar por uma segunda chamada de modelo) — é exatamente o problema que "assíncrono" no pedido original resolve.

## 2. `forget_preference` reaproveita a tool `forget_fact` já existente

**Decision**: nenhuma tool nova é criada. A tool `forget_fact` (`src/memory/memory-store.ts`, `createMemoryTools`, `007-semantic-memory`) já remove qualquer fato da pessoa por descrição, independente de como ele foi originalmente registrado — e um fato gravado pelo refletor (via `store.remember`) fica na mesma tabela `memories`, indistinguível de um fato ensinado manualmente. `forget_preference`, citado no pedido original, é portanto o mesmo comportamento que `forget_fact` já entrega; a spec (Assumptions) já documenta essa decisão.

**Rationale**: criar uma segunda tool (`forget_preference`) com semântica idêntica a `forget_fact` duplicaria código e daria ao modelo duas ferramentas concorrentes para a mesma ação — más para o modelo decidir entre si, sem nenhum ganho real (Princípio VIII, Pequeno e Reversível). A US2 do spec 008 (desfazer um aprendizado automático) já é satisfeita pela infraestrutura de `007` sem nenhuma mudança de código; o trabalho desta feature nesse ponto é só validar isso com um teste dedicado (`learning-reflector.test.ts`), não construir nada novo.

**Alternatives considered**: renomear `forget_fact` para `forget_preference`. Rejeitada: mudaria o contrato de tool já em produção desde `007` sem necessidade — quebra reversibilidade e não adiciona valor, já que o nome exposto ao modelo não faz diferença para quem usa o `/chat` via HTTP.

## 3. Tratamento de falha do refletor

**Decision**: dupla camada de proteção. `reflectAndRemember` (`src/memory/learning-reflector.ts`) nunca rejeita — qualquer erro de `distillFn` ou de `store.remember` é capturado internamente e só gera um `console.error` de diagnóstico (FR-006). O controller HTTP, além disso, encadeia um `.catch(() => {})` no-op sobre a chamada não aguardada, como cinto de segurança contra qualquer implementação injetada (ex.: em teste) que não respeite esse contrato.

**Rationale**: FR-006 exige que a falha seja absorvida "do ponto de vista da pessoa usuária" — como a chamada já não é aguardada (item 1), uma rejeição não tratada se tornaria um `unhandledRejection` do processo Node, não um erro HTTP visível; ainda assim, deixar isso acontecer seria um comportamento frágil e ruidoso (log de crash, possível encerramento do processo dependendo da configuração). A dupla camada custa uma linha e elimina essa classe de falha por completo.

## 4. Estratégia de teste do `withStructuredOutput` real

**Decision**: mesmo padrão de `src/agents/reflection.ts`/`reflection.test.ts` — a função que chama `createModel().withStructuredOutput(...)` (`distillLearning`) não é exercitada diretamente por teste automatizado (não há mock de rede/API para o modelo de raciocínio em nenhum teste do projeto hoje); a lógica de decisão (`reflectAndRemember`: "se `hasLearning` e houver `fact`, chama `store.remember`; senão, não") é extraída como função separada que recebe `distillFn` por injeção de dependência (mesmo padrão de `runReflectionLoop` recebendo `critiqueFn`) e é testada com fakes determinísticos, sem rede.

**Rationale**: o projeto já tem essa convenção estabelecida (`resolveStrategy`, `conversationStore`, `memoryStore`, `embedFn` — todos injetáveis, todos com produção real + fake em teste) — manter `npm test` 100% determinístico e offline sem exigir `OPENROUTER_API_KEY` em CI, exatamente como já acontece hoje.

**Alternatives considered**: gravar cassettes/fixtures de resposta real do modelo (VCR-style). Rejeitada: nenhuma outra feature do projeto faz isso (nem `007` para embeddings determinísticos, nem `reflection.ts` para o crítico) — introduziria uma técnica nova sem precedente para um ganho marginal, já que a lógica de decisão em si é trivial e totalmente coberta por fakes.

## 5. Nenhuma tabela nova

**Decision**: a "Reflexão de aprendizado" (resultado `{ hasLearning, fact? }`) é um valor transiente, nunca persistido por si só — o único efeito de armazenamento continua sendo `store.remember(userId, fact)`, já existente e já com sua própria tabela (`memories`, `007-semantic-memory`).

**Rationale**: persistir a reflexão em si (ex.: para auditoria de "o que foi analisado e descartado") não foi pedido pela spec nem pelo pedido original — adicionar isso seria escopo não solicitado (evitado por padrão neste projeto, Princípio VIII).

## 6. Critério de "fato durável" vs. "pedido pontual" vs. "segredo"

**Decision**: delegado inteiramente ao modelo via prompt de sistema de `distillLearning` — nenhuma heurística determinística (regex, lista de palavras-chave) no domínio para classificar segredos ou distinguir fato de pedido pontual.

**Rationale**: mesma linha já registrada nas Assumptions do `spec.md` — uma lista fixa de padrões de segredo seria incompleta por natureza (token/chave têm formatos arbitrários) e daria falsa sensação de segurança; o julgamento do próprio modelo, guiado por um prompt explícito ("nunca pedido pontual, nunca segredo"), é a mesma abordagem que o pedido original descreve (`withStructuredOutput({ hasLearning, fact })` decidindo, não uma função pura verificando).

**Alternatives considered**: complementar o julgamento do modelo com uma lista de regex para credenciais óbvias (ex.: `sk-`, `Bearer `, sequências longas de caracteres aleatórios) como camada extra de segurança. Considerada, mas não adotada nesta primeira versão: adicionaria complexidade e uma segunda fonte de verdade sem pedido explícito da spec; fica registrada aqui como possível trabalho futuro caso SC-002 não seja atingido na prática.
