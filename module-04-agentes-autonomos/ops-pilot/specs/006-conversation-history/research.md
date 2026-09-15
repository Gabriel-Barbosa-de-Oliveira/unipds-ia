# Research: Conversa Persistente

## 1. Como o histórico chega ao raciocínio do agente

**Decision**: o histórico é composto, no controller HTTP, em uma única string que substitui o `input` já hoje aceito por `ReasoningStrategy.run(input, options)` — a assinatura da interface (`src/agents/types.ts`) não muda.

**Rationale**: `input` já é uma string opaca hoje, e ambas as estratégias já a tratam como tal em múltiplos pontos: `react.ts` a injeta como `{role:"user", content: input}`; `plan-and-execute.ts` a carrega em `state.input` e a interpola diretamente no prompt do planner e do replanner (`buildReplanPrompt`, `Pedido original do plantonista: ${state.input}`); `reflection.ts` a reencaminha e a reescreve em `buildRetryInput` sem nunca inspecionar sua estrutura. Compor o histórico como texto antes de chamar `strategy.run(...)` (exatamente como o pedido original descreve: "12 últimas mensagens no prompt **via composição**") entrega o comportamento pedido sem tocar `react.ts`, `plan-and-execute.ts`, `reflection.ts` ou `message-trace.ts` — zero risco de regressão nas 3 features já implementadas sobre esses arquivos (001–003), e o menor diff possível (Princípio VIII).

**Alternatives considered**: estender `RunOptions`/a assinatura de `run()` para aceitar uma lista de mensagens (`{role, content}[]`) e fazer cada estratégia semear `agent.stream({messages: [...history, novaMensagem]})` nativamente. Rejeitada: exigiria mudanças estruturais em pelo menos 3 arquivos (`react.ts`, `plan-and-execute.ts` — que não usa `agent.stream` com uma lista de mensagens crescente, e sim um único `state.input` textual reinterpolado a cada replanejamento — e `reflection.ts`, cujo `buildRetryInput` opera sobre uma string), para um ganho (roles nativos por mensagem na chamada ao modelo) que a spec não pede.

## 2. Onde vive a persistência de conversas

**Decision**: uma classe nova, `SqliteConversationStore` (`src/store/sqlite-conversation-store.ts`), com duas tabelas próprias (`conversations`, `messages`) e conexão `node:sqlite` (`DatabaseSync`) independente da de `SqliteOpsStore` — mesmo arquivo (`OPSPILOT_DB`), duas conexões, mesmo padrão de inicialização lazy da DDL (`004-ops-persistence`).

**Rationale**: conversa/histórico de chat é um bounded context distinto de dados operacionais (serviços, alertas, incidentes, runbooks) — misturar as duas na mesma classe infla `SqliteOpsStore` com uma responsabilidade que nenhuma tool de domínio usa, e obrigaria a tocar um arquivo já estável de uma feature anterior. Duas conexões `DatabaseSync` sobre o mesmo arquivo SQLite é seguro (SQLite suporta múltiplas conexões de processo únicas com locking próprio) e mantém `sqlite-ops-store.ts` inteiramente inalterado.

**Alternatives considered**: adicionar `conversations`/`messages` à DDL de `SqliteOpsStore` e os métodos `create`/`append`/`lastMessages` à mesma classe. Rejeitada: acopla dois domínios sem relação (nenhuma tool de ops referencia conversa, nenhuma operação de conversa referencia serviço/incidente), e obriga qualquer teste futuro de `SqliteOpsStore` a carregar um schema maior sem motivo.

## 3. O que é gravado como mensagem

**Decision**: cada turno grava a mensagem **crua** enviada pela pessoa de plantão (`parsed.data.message`, texto original) e a resposta final do copiloto (`result.answer`) — nunca o prompt já composto com histórico.

**Rationale**: se o texto composto (histórico + mensagem atual) fosse o que se grava, cada novo turno reincorporaria todo o histórico anterior dentro da própria mensagem armazenada, e a próxima composição duplicaria esse conteúdo — crescimento exponencial do texto armazenado e do prompt enviado ao modelo a cada turno. Gravar apenas o turno literal mantém `lastMessages(conversationId, 12)` como uma janela linear real sobre o histórico.

## 4. Identificador de conversa desconhecido

**Decision**: `conversationStore.lastMessages(conversationId, limit)` lança `ConversationNotFoundError` (nova classe em `src/domain/errors.ts`) quando `conversationId` não corresponde a nenhuma conversa existente; o controller propaga isso para `errorMiddleware`, que responde `404 { error: "conversation_not_found", conversationId }`.

**Rationale**: mesmo padrão já usado para `ServiceNotFoundError`/`IncidentNotFoundError` — "referenciar algo que não existe" é um erro de domínio explícito, nunca uma string solta ou um estado ambíguo. A diferença frente a esses dois é que a resolução da conversa acontece **antes** de qualquer chamada ao agente (não dentro de uma tool), então este é o primeiro erro de domínio deste projeto que precisa de tradução http explícita direto no `errorMiddleware` de `src/http/server.ts`, e não apenas dentro da tool que o lança.

**Alternatives considered**: tratar um `conversationId` desconhecido como início silencioso de uma conversa nova sob aquele mesmo id. Rejeitada: mascara um erro provável de quem chama (id errado, truncado, ou de um ambiente diferente) atrás de um comportamento que parece bem-sucedido — contraria a preferência do projeto por erros de domínio explícitos (Princípio III) e foi resolvida como assumption na spec (FR-007).

## 5. Onde mora `historyMessages`

**Decision**: `historyMessages` é calculado no controller (`length` do array retornado por `lastMessages`) e mesclado no objeto `metrics` da resposta HTTP — **não** é adicionado a `Metrics`/`RunResult` (`src/agents/types.ts`).

**Rationale**: `RunResult`/`Metrics` são o contrato usado por toda estratégia, por `bench.ts`, `arena.ts` e pelos testes das 3 features anteriores — nenhum desses consumidores tem noção de conversa. Mesclar o campo só na borda HTTP (onde a conversa de fato existe) evita qualquer mudança nesses arquivos e mantém `ReasoningStrategy` com a mesma assinatura de resultado desde a feature 001.

## 6. Conversa criada antes da execução da estratégia

**Decision**: quando nenhum `conversationId` é informado, o controller chama `conversationStore.create()` **antes** de rodar a estratégia, para já ter um identificador estável a devolver e a usar no `append` final.

**Consequência aceita**: se a execução da estratégia falhar (500/504), a conversa já foi criada no armazenamento, mas nenhuma mensagem é anexada a ela — uma linha "órfã" (conversa vazia) permanece no banco. Isso é inofensivo: a resposta de erro nunca inclui esse `conversationId` (o contrato de erro de `003-chat-endpoint` não é alterado por esta feature), então ninguém consegue referenciá-la depois; não há vazamento de dado sensível, apenas uma linha vazia sem custo prático. Adicionar lógica de rollback (apagar a conversa se a estratégia falhar) foi descartado por adicionar complexidade sem benefício observável para quem usa o sistema.

## 7. Ordenação das mensagens

**Decision**: a tabela `messages` usa `id INTEGER PRIMARY KEY AUTOINCREMENT` como chave de ordenação (não `created_at`) — `lastMessages` ordena por esse id.

**Rationale**: um único turno grava a mensagem do usuário e a resposta do copiloto em sequência, potencialmente no mesmo milissegundo — ordenar por timestamp arriscaria empate e inverteria a ordem cronológica real (resposta antes da pergunta). `AUTOINCREMENT` garante ordem estritamente crescente e igual à ordem de inserção, sem esse risco.

## 8. Geração do identificador de conversa

**Decision**: `crypto.randomUUID()`, mesmo mecanismo já usado por `SqliteOpsStore.openIncident` para o `id` de um incidente.

**Rationale**: consistência com o padrão já estabelecido no projeto; nenhuma dependência nova.
