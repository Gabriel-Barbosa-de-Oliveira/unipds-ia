# Research: Modo Equipe

Decisões da Fase 0 do `/speckit.plan`. Cada item segue o formato Decisão / Racional / Alternativas.

## 1. Onde a equipe entra no grafo de produção

**Decisão**: a equipe é uma `ReasoningStrategy` (`createTeamStrategy`, em `src/team/`) com um `StateGraph` próprio por dentro. O grafo de produção (012) ganha a rota e o nó `team`, ligados como as outras estratégias: `roteador → team → resposta`. O nó usa o `strategyNode("team")` que já existe, que carimba `node: "team"` e reindexa `at`.

**Racional**: reaproveita tudo o que o grafo já faz por estratégia: contexto, métricas somadas, persistência no nó `resposta`, timeout e 202. As outras rotas não mudam (FR-017).

**Alternativas**: pôr supervisor e papéis como nós soltos no grafo de produção. Isso misturaria o estado da equipe com o do grafo, e `tagTrace` e as métricas teriam que mudar.

## 2. Supervisor

**Decisão**:
- Schema: `SupervisorDecisionSchema = z.object({ next: z.enum(["analista", "planejador", "executor", "FINISH"]), brief: z.string().trim().min(1) })`.
- Chamada: `createModel((m) => m.withStructuredOutput(SupervisorDecisionSchema))`, o mesmo padrão do roteador (013), resiliente e com reserva.
- O supervisor recebe o pedido e o quadro renderizado, e não as mensagens brutas dos papéis.
- **Com `next: "FINISH"`, o `brief` é a resposta final** para a pessoa. Isso evita uma chamada extra e cumpre a premissa "o supervisor redige a resposta final".
- A decisão crua passa por `resolveSupervisorDecision(raw, turn, cap)`, que é pura e devolve uma de três coisas: `{ kind: "route", to, brief }`, `{ kind: "finish", answer }` ou `{ kind: "abort", reason }`. Os casos de `abort` são decisão inválida, exceção do modelo e teto atingido (FR-005).

**Racional**: o `{ next, brief }` foi pedido pela pessoa usuária. Validar numa função pura deixa o loop testável sem modelo.

## 3. Quadro compartilhado (blackboard)

**Decisão**: o tipo `Blackboard` fica em `src/team/blackboard.ts`, com funções puras e imutáveis:

```ts
interface Blackboard {
  request: string;
  facts: { statement: string; source: AnalystSource }[];  // só o analista escreve
  plan: string[] | null;                                   // só o planejador escreve
  outcomes: { summary: string; proposal: boolean }[];      // só o executor escreve
  handoffs: { from: "supervisor"; to: TeamRole | "fim"; brief: string }[];
}
```

As funções `addFacts`, `setPlan`, `addOutcome` e `addHandoff` devolvem cópias. `renderBlackboard(bb)` gera o texto em markdown que vai no prompt do supervisor e dos papéis. O quadro fica numa chave do estado do `StateGraph` da equipe, com reducer de substituição.

**Racional**: cada papel só tem uma função de escrita. O tipo impede, por exemplo, que o analista escreva um plano.

## 4. Analista: só leitura e "não propõe" por construção

**Decisão**: o analista roda em dois passos.
1. Um `createReactAgent` com **apenas** `list_alerts`, `list_incidents` e `consultar_runbook`, recebendo o brief do supervisor e o quadro.
2. Uma extração estruturada com `createModel((m) => m.withStructuredOutput(AnalystReportSchema))` sobre a transcrição do passo 1, onde `AnalystReportSchema = z.object({ facts: z.array(z.object({ statement, source: z.enum([...as 3 tools, "pedido"]) })).max(20) })`.

**Só os `facts` vão para o quadro.** O texto livre do agente (onde caberia uma recomendação) nunca chega ao planejador nem ao supervisor (FR-007).

**Racional**: o `responseFormat` do `createReactAgent` 0.2.74 exige um modelo com `.withStructuredOutput()`, e o modelo resiliente (`toolCallingModel`, um `RunnableBinding` sobre `withFallbacks`) não tem esse método (`react_agent_executor.d.ts:93`). A extração separada usa a mesma fábrica do roteador.

**Alternativas**: confiar só no prompt para "não propor" (não é garantido pelo sistema, e a spec pede garantia); filtrar a prosa por regex (frágil).

## 5. Planejador: sem ferramentas

**Decisão**: uma chamada `createModel((m) => m.withStructuredOutput(PlanSchema))`, com `PlanSchema = z.object({ steps: z.array(z.string().min(1)).min(1).max(8) })`, sobre o brief e o quadro. O modelo não recebe nenhum `bindTools` (FR-008). O resultado vira `setPlan`.

## 6. Executor: só incidentes, sem bypass

**Decisão**:
- `createReactAgent` com **apenas** `open_incident` e `resolve_incident`, recebendo o brief e o quadro.
- As ferramentas vêm das `baseTools` que o `/chat` já entrega, criadas por `createGatedOpsTools` (015).
- **Garantia de não-bypass:** `createGatedOpsTools` registra as instâncias com porta num `WeakSet`, e `approval-gate.ts` exporta `isApprovalGated(tool)`. Na construção, `createTeamStrategy` falha (`TeamToolsNotGatedError`, erro de domínio) se as ferramentas `open_incident` e `resolve_incident` recebidas não forem as instâncias com porta. Assim nenhuma composição, nem uma futura, entrega ao executor uma ferramenta que executa direto (FR-010).
- O resultado do turno vira `addOutcome({ summary, proposal })`. `proposal` é `true` quando o trace do turno tem uma observação `status: "awaiting_approval"`, detectado por função pura.
- **Depois de uma proposta, a equipe encerra:** o supervisor não é chamado de novo, e o loop registra o handoff `→ fim` com brief "ação aguardando aprovação". O `/chat` já responde 202 a partir do `gate`.

**Racional**: a porta da 015 já resolve a aprovação. O que faltava era impedir que a equipe recebesse ferramentas sem porta, e a checagem por identidade de instância resolve isso sem flags que poderiam ser forjadas.

## 7. Loop, teto e encerramento

**Decisão**:
- Estado da equipe: `{ blackboard, turn, trace: TeamTraceEvent[], decision }`.
- Arestas: `START → supervisor`, `supervisor → (analista | planejador | executor | END)` e cada papel `→ supervisor`, exceto o executor depois de uma proposta, que vai para `END`.
- **Teto:** `TEAM_MAX_TURNS = 6` turnos de papel. Na 7ª decisão que não seja `FINISH`, o resultado é `abort("teto de 6 passagens atingido")`.
- **Abort:** registra o handoff `→ fim` com o motivo, e a resposta vem de `fallbackAnswer(blackboard, reason)`, que é pura e resume fatos, plano e resultados.
- `recursionLimit` do `StateGraph`: `2 * TEAM_MAX_TURNS + 4`, como segunda proteção contra loop.

## 8. Evento `handoff` e papel no trace

**Decisão**:
- `TraceEvent` ganha a variante `{ type: "handoff"; at; from: "supervisor"; to: TeamRole | "fim"; brief: string }`.
- Todos os eventos ganham o campo opcional `role?: TeamRole | "supervisor"`, assim como `node?` já existe.
- Dentro da equipe, todo evento leva `role`: os do analista `role: "analista"` e assim por diante. A resposta final é `{ type: "answer", role: "supervisor" }`.
- Os eventos dos papéis vêm de `messagesToTrace`, que não muda. O loop só carimba `role` e reindexa `at` por função pura (`tagRole`).

**Persistência (014)**: `trace_events.type` e `node` não têm CHECK, e o `payload_json` guarda o evento completo, inclusive `role`, `from`, `to` e `brief`. O `restoreTrace` devolve um trace idêntico ao original sem mudança (FR-013).

**Logs (FR-015)**: `traceToLogEvents` passa a mapear `handoff` para `{ event: "team.handoff", requestId, node, position, from, to }`, sem `brief`. Um teste-âncora garante que o brief nunca aparece na linha de log.

## 9. Rota `team` e migração do SQLite

**Decisão**:
- `ROUTE_NAMES` ganha `"team"`. O `routeSchema`, o `GraphNode` e a tabela do roteador acompanham.
- `ROUTE_TABLE` ganha a linha `team`: "Investigar e agir de forma coordenada (levantar fatos, planejar, executar)", com exemplos.
- `ROUTE_ALIASES` ganha `team` e `equipe`. `strategyForRoute("team")` chama `resolve("team", false, extraTools, baseTools)`. A flag `reflect` é ignorada, como na rota `reflect`.
- **Migração:** a coluna `requests.route` tem `CHECK (route IN ('react','planExecute','reflect'))`. Em bancos existentes, `CREATE TABLE IF NOT EXISTS` não muda o CHECK, e gravar `team` falharia (`persistence.failed`). O `SqliteRequestStore` ganha `migrateRouteCheck(db)`, idempotente: se o `sql` de `requests` em `sqlite_master` não contiver `'team'`, reconstrói a tabela pelo procedimento oficial do SQLite (`PRAGMA foreign_keys=OFF` → `BEGIN` → `CREATE requests_new` → `INSERT … SELECT` → `DROP requests` → `ALTER … RENAME` → recriar índices → `PRAGMA foreign_key_check` → `COMMIT` → `foreign_keys=ON`). O `node:sqlite` liga chaves estrangeiras por padrão, e `trace_events` referencia `requests`, por isso o `foreign_keys=OFF` é obrigatório. O DDL novo já nasce com `'team'`.

**Racional**: sem a migração, toda execução da equipe perderia o registro (014) em bancos antigos, como o `data/opspilot.db` atual.

**Alternativas**: tirar o CHECK (perde a validação no banco e exige a mesma reconstrução); criar uma coluna nova (deixaria dados duplicados).

## 10. Métricas (FR-018)

**Decisão**: o `LlmCallCounter`, o `UsageCollector` e o `ModelUsageTracker` são criados uma vez por execução da equipe e passados em `callbacks` para todas as chamadas (supervisor, agentes e extrações). `buildMetrics` e `summarizeModelUsage` produzem as métricas como nas outras estratégias. O `modelUsed` é o da última resposta, e as trocas de modelo viram eventos `fallback` (013).

## 11. Testabilidade sem rede

**Decisão**: `createTeamStrategy(tools, deps?)` aceita `deps` injetáveis: `decide` (supervisor), `runAnalyst`, `runPlanner` e `runExecutor`, cada um com uma assinatura pura de entrada e saída sobre o quadro e o brief. O padrão usa os modelos reais. Os testes injetam fakes determinísticos e cobrem: sequência de turnos, teto, decisão inválida, encerramento por proposta, papéis com as ferramentas certas e a falha com ferramentas sem porta.

## 12. War room

**Decisão**:
- `api-schemas.ts` ganha o ramo `handoff` e `role` opcional na base.
- `trace-view.ts`: o `handoff` usa o rótulo "Passagem", o ícone `handoff` e o corpo `{ kind: "handoff", from, to, brief }`, com os papéis traduzidos ("Analista", "Planejador", "Executor", "Supervisor", "Fim"). Cada `TraceView` ganha `role: string | null`.
- `TraceEventItem` mostra um badge do papel ao lado do badge do nó e renderiza a passagem como `Supervisor → Analista` mais a instrução.
- `tokens.css` ganha `--trace-handoff` nos dois temas, com contraste ≥ 4.5:1 verificado pelo mesmo script.
- O teste de contrato (`web-contract.test.ts`) passa a cobrir um 200 da equipe com `handoff`.

## 13. Constitution

Sem amendment. Tudo fica dentro da stack existente (LangGraph, zod e SQLite). O Princípio VI é reforçado pela checagem de ferramentas com porta (item 6).
