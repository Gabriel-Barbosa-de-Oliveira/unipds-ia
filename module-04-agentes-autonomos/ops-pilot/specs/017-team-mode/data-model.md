# Data Model: Modo Equipe

## Papéis

| Papel | `TeamRole` | Ferramentas (por nome) | Escreve no quadro | Saída estruturada |
|---|---|---|---|---|
| Supervisor | `"supervisor"`, só em `role` | nenhuma | `handoffs` | `{ next: "analista" \| "planejador" \| "executor" \| "FINISH", brief: string }` |
| Analista | `"analista"` | `list_alerts`, `list_incidents`, `consultar_runbook` | `facts` | `{ facts: { statement: string; source: "list_alerts" \| "list_incidents" \| "consultar_runbook" \| "pedido" }[] }` (máx. 20) |
| Planejador | `"planejador"` | nenhuma | `plan` | `{ steps: string[] }` (1–8) |
| Executor | `"executor"` | `open_incident`, `resolve_incident`, **só as instâncias com porta** | `outcomes` | — (texto + detecção de proposta) |

As listas de ferramentas ficam em `TEAM_ROLE_TOOLS`, uma constante. `selectRoleTools(all, role)` é pura e falha se faltar alguma ferramenta da lista.

## Blackboard

```ts
interface Blackboard {
  request: string;
  facts: Fact[];                 // Fact = { statement; source }
  plan: string[] | null;
  outcomes: Outcome[];           // Outcome = { summary; proposal: boolean }
  handoffs: Handoff[];           // Handoff = { from: "supervisor"; to: TeamRole | "fim"; brief }
}
```

**Regras**:
- `facts` só cresce. Fatos duplicados (mesmo `statement` e `source`) são ignorados.
- `plan` é substituído a cada turno do planejador.
- Um `outcome` com `proposal: true` encerra a equipe.
- `renderBlackboard` mostra até os últimos 20 fatos e trunca cada item em 500 caracteres, para limitar o prompt.

## Estado da equipe (`StateGraph`)

| Campo | Tipo | Reducer |
|---|---|---|
| `blackboard` | `Blackboard` | substituição |
| `turn` | `number` (turnos de papel já executados) | substituição |
| `pending` | `{ to: TeamRole; brief: string } \| { finish: string }` | substituição |
| `trace` | `TraceEvent[]`, já com `role` | concatenação |

**Transições**:

```text
START → supervisor
supervisor ─resolveSupervisorDecision─┬─ route(to)  → <papel> → supervisor
                                      ├─ finish     → END (answer = brief)
                                      └─ abort      → END (answer = fallbackAnswer)
executor com proposta → END (handoff → fim "ação aguardando aprovação")
turn == TEAM_MAX_TURNS (6) e decisão ≠ FINISH → abort("teto")
```

## Trace (`src/agents/types.ts`)

- Variante nova: `{ type: "handoff"; at: number; from: "supervisor"; to: TeamRole | "fim"; brief: string }`.
- Campo opcional novo em todos os eventos: `role?: TeamRole | "supervisor"`.
- `GraphNode` e `RouteName` ganham `"team"`.

## Erros de domínio

| Classe | Quando | Efeito |
|---|---|---|
| `TeamToolsNotGatedError` | `createTeamStrategy` recebe `open_incident`/`resolve_incident` sem porta | Falha ao montar a estratégia, ou seja, erro de programação. Pelo `/chat` vira 500, nunca uma execução sem porta |

## SQLite

- `requests.route`: o CHECK passa a aceitar `'team'`. `migrateRouteCheck` reconstrói a tabela em bancos antigos e é idempotente (research.md item 9).
- `trace_events`: não muda (`payload_json` guarda `role`, `from`, `to` e `brief`).

## Log (014)

`LogEvent` novo: `{ event: "team.handoff"; requestId; node: GraphNode | null; position; from: "supervisor"; to: TeamRole | "fim" }`, com nível `info` e **sem `brief`**.
