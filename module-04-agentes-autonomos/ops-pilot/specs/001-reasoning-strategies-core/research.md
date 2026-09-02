# Phase 0 Research: Núcleo de Raciocínio (Reasoning Strategies Core)

## 1. Duplo backend de store: in-memory vs. Sequelize/MySQL

**Decision**: Definir uma única interface de repositório (`OpsStoreRepository`: `listAlerts(status?)`, `openIncident(input)`, `resolveIncident(id)`) com duas implementações atrás dela — `ops-store.memory.ts` (estado mantido em processo, populado pelo seed canônico) e `ops-store.sequelize.ts` (modelos Sequelize sobre MySQL). O adaptador in-memory é o padrão usado por `src/agents/tools.ts`, pela arena e pelos testes; o adaptador Sequelize/MySQL fica disponível para quando o store precisa persistir de fato (ex.: futura integração com `src/index.ts`/API), alimentado pelo mesmo dataset via `src/scripts/seed.ts`.

**Rationale**: O pedido original contém duas afirmações aparentemente conflitantes — "ferramentas mock... sobre um store in-memory pré-populado" e "banco mysql, utilizando sequelize". As duas são necessidades reais e atuais, não hipotéticas: (a) FR-014 exige que a lógica do store seja testável sem rede, o que um backend MySQL real inviabiliza em CI/local sem banco disponível; (b) a constitution do projeto marca Sequelize + MySQL como **Stack Tecnológica Obrigatória**, não opcional. Uma interface fina com duas implementações resolve ambas sem duplicar regra de negócio (que fica em `src/domain/ops-store.ts`, puro, independente do adaptador).

**Alternatives considered**:
- *Só in-memory, sem Sequelize*: mais simples, mas viola a stack obrigatória da constitution e ignora a instrução explícita "banco mysql, utilizando sequelize".
- *Só Sequelize/MySQL, sem in-memory*: exigiria um MySQL disponível para qualquer teste ou execução da arena, quebrando FR-014 (testes determinísticos sem rede) e tornando a arena (ferramenta de avaliação rápida) dependente de infraestrutura.
- *Sequelize com SQLite in-memory para testes*: ainda seria IO real (driver de banco, mesmo que em memória) e adicionaria uma dependência nova (`sqlite3`) fora da stack já declarada; rejeitado por complexidade extra sem necessidade.

## 2. Fábrica única de modelo (OpenRouter)

**Decision**: `src/agents/model.ts` exporta uma única função `createReasoningModel()` que instancia `ChatOpenAI` (`@langchain/openai`) com `configuration.baseURL` apontando para o endpoint do OpenRouter, `apiKey: process.env.OPENROUTER_API_KEY`, `model: process.env.OPENROUTER_MODEL` e `temperature: 0`. Nenhuma outra parte do código instancia um cliente de modelo diretamente — todas as estratégias (`react.ts`, `plan-and-execute.ts`) recebem o modelo desta fábrica.

**Rationale**: OpenRouter expõe uma API compatível com a API da OpenAI; `@langchain/openai` (já presente nas dependências) suporta customizar `baseURL`, o que evita adicionar um cliente HTTP extra. Uma fábrica única garante que `temperature: 0` (determinismo) e a leitura das duas env vars fiquem em um único ponto, testável/mockável ao trocar a estratégia por um dublê nos testes que não devem tocar rede.

**Alternatives considered**:
- *Instanciar o modelo dentro de cada estratégia*: duplica a leitura de env vars e o risco de divergência de configuração (ex.: uma estratégia esquecer `temperature: 0`); rejeitado.
- *Cliente HTTP customizado para OpenRouter*: reinventa o que `@langchain/openai` já oferece via `baseURL`; rejeitado por complexidade desnecessária.

## 3. Captura de trace na estratégia ReAct

**Decision**: Usar `createReactAgent` de `@langchain/langgraph/prebuilt` com as três tools, invocado em modo de streaming de eventos (`stream`/`streamEvents`) para capturar, na ordem em que ocorrem, mensagens do modelo (mapeadas para eventos `thought`/`answer`), chamadas de tool (`action`, com nome e args) e resultados de tool (`observation`). Um adaptador em `src/agents/react.ts` traduz os eventos nativos do LangGraph para a união `TraceEvent` definida em `src/agents/types.ts`, contando uma chamada de LLM a cada invocação do modelo observada no stream.

**Rationale**: O agente ReAct pré-construído do LangGraph já expõe o necessário via streaming de eventos, evitando reimplementar o loop ReAct manualmente (o pedido original pede explicitamente o agente pré-construído). Mapear para a união `TraceEvent` comum mantém a estratégia ReAct plugável na mesma interface `ReasoningStrategy` usada pelo Plan-and-Execute.

**Alternatives considered**:
- *Reimplementar o loop ReAct manualmente*: contradiz o pedido explícito de usar o agente pré-construído; mais código para manter.
- *Inspecionar apenas o estado final (sem streaming)*: mais simples, mas perde a ordem/latência por passo dentro do trace, enfraquecendo a auditabilidade exigida por FR-002/FR-013.

## 4. Grafo Plan-and-Execute

**Decision**: `src/agents/plan-and-execute.ts` implementa um `StateGraph` do LangGraph com três nós — `planner` (saída estruturada: lista ordenada de passos, via `withStructuredOutput`), `executor` (executa o próximo passo pendente usando as tools disponíveis, um passo por chamada) e `replanner` (reavalia os passos restantes após cada execução; decide finalizar quando não resta nenhum passo ou decide gerar uma resposta final). Um contador de passos executados no estado do grafo encerra a execução ao atingir 8, mesmo que o replanner ainda proponha passos.

**Rationale**: Segue o padrão de referência "Plan-and-Execute" do LangGraph (planner → executor → replanner em loop), que já resolve replanejamento incremental sem reconstruir o plano do zero a cada passo. O limite de 8 passos é aplicado no grafo (não apenas como uma sugestão ao replanner) para satisfazer FR-005 mesmo se o modelo tentar propor mais passos.

**Alternatives considered**:
- *Plano fixo executado sem replanejamento*: mais simples, mas não atende ao requisito explícito de revisão do plano após cada passo (User Story de comparação depende de estratégias com comportamentos distintos).
- *Sem limite físico no grafo, apenas instrução via prompt*: risco de o modelo ignorar a instrução e estourar o limite; rejeitado por não satisfazer FR-005/FR-006 de forma confiável.

## 5. Contagem de chamadas de LLM e latência

**Decision**: Cada estratégia mede `metrics.llmCalls` incrementando um contador toda vez que o modelo da fábrica única é invocado (interceptado via callback/wrapper compartilhado, não reimplementado em cada estratégia) e `metrics.latencyMs` como o tempo total decorrido entre o início e o fim de `run(input)` (via `Date.now()`/`performance.now()` capturado no ponto de entrada/saída comum da interface `ReasoningStrategy`).

**Rationale**: Medir no ponto de entrada/saída comum (e não dentro de cada estratégia individualmente) garante que a definição de "chamada de LLM" e "latência" seja idêntica entre ReAct e Plan-and-Execute — pré-requisito para a comparação da arena (FR-003, FR-011, User Story 2) ser justa.

**Alternatives considered**:
- *Cada estratégia calcula suas próprias métricas de forma independente*: risco de definições divergentes (ex.: uma contar apenas chamadas ao planner, outra contar todas); rejeitado por comprometer a comparação lado a lado.

## 6. Contrato da CLI da arena

**Decision**: `src/arena.ts` valida seus argumentos (`--strategies` uma lista separada por vírgula dentre `react`/`plan-and-execute`, `--max-iterations` um inteiro positivo opcional, e o texto de entrada) com um schema zod antes de rodar qualquer estratégia. Para cada estratégia selecionada, imprime um bloco identificado pelo nome da estratégia contendo o trace formatado passo a passo e as métricas; ao final, imprime um resumo comparativo (nome, nº de chamadas de LLM, latência) quando duas ou mais estratégias são executadas.

**Rationale**: A CLI é uma fronteira externa (Principle II da constitution) — precisa de validação zod como qualquer entrada HTTP. Blocos identificados por estratégia e um resumo final atendem diretamente à User Story 2 (comparação lado a lado sem precisar re-executar nada).

**Alternatives considered**:
- *Saída em JSON bruto apenas*: mais fácil de parsear programaticamente, mas pior para o caso de uso descrito ("imprime traces e métricas") de leitura humana direta no terminal; pode ser adicionado depois como flag `--json` se necessário, fora do escopo atual.

## 7. Estratégia de testes determinísticos sem rede

**Decision**: Testes cobrem exclusivamente lógica pura e sem IO: (a) `src/domain/ops-store.ts` — transições de estado do store (listar por status, abrir incidente válido/inválido, resolver incidente existente/inexistente) operando sobre um estado in-memory construído no próprio teste; (b) `src/agents/trace.ts` — formatação/serialização de uma sequência de `TraceEvent` fornecida como fixture, sem invocar nenhuma estratégia real nem o adaptador Sequelize. Nenhum teste instancia o modelo da fábrica única nem abre conexão MySQL.

**Rationale**: Atende literalmente FR-014 ("Testes: store e formatação de trace, determinísticos, sem rede") e à Principle V da constitution (nenhuma lógica nova sem teste), mantendo o `npm test` executável em CI sem credenciais de OpenRouter nem MySQL disponíveis.

**Alternatives considered**:
- *Testes de integração ponta a ponta contra OpenRouter/MySQL reais*: úteis eventualmente, mas fora do escopo desta feature (que pede explicitamente testes sem rede); podem ser propostos como feature separada (ex.: smoke tests de CI) no futuro.
