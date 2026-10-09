import type { Callbacks } from "@langchain/core/callbacks/manager";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";

import type { HandoffTarget, TeamRole, TraceEvent } from "../agents/types.ts";
import {
  addFacts,
  addHandoff,
  addOutcome,
  createBlackboard,
  fallbackAnswer,
  renderBlackboard,
  setPlan,
  type Blackboard,
  type Fact,
} from "./blackboard.ts";
import { buildSupervisorMessages, resolveSupervisorDecision, TEAM_MAX_TURNS } from "./supervisor.ts";

/** O que cada papel recebe: a instrução do supervisor e o quadro como está. */
export interface RoleInput {
  request: string;
  brief: string;
  blackboard: Blackboard;
  callbacks?: Callbacks;
}

/**
 * Cada papel devolve só a sua contribuição; é o grafo quem escreve no quadro, então nenhum papel
 * consegue escrever no campo de outro (FR-007).
 */
export interface TeamDeps {
  decide(messages: [string, string][], callbacks?: Callbacks): Promise<unknown>;
  runAnalyst(input: RoleInput): Promise<{ trace: TraceEvent[]; facts: Fact[] }>;
  runPlanner(input: RoleInput): Promise<{ trace: TraceEvent[]; steps: string[] }>;
  runExecutor(input: RoleInput): Promise<{ trace: TraceEvent[]; summary: string }>;
}

export interface TeamRunResult {
  answer: string;
  trace: TraceEvent[];
  blackboard: Blackboard;
  /** O executor propôs uma ação, que aguarda aprovação humana (o /chat responde 202). */
  proposed: boolean;
}

export const PROPOSAL_BRIEF = "Ação aguardando aprovação humana; a equipe encerra aqui.";
export const PROPOSAL_ANSWER = "A ação proposta pelo executor aguarda aprovação humana.";

/** Carimba o papel e reindexa `at` a partir de `offset`. Pura: devolve cópias. */
export function tagRole(events: readonly TraceEvent[], role: TeamRole, offset: number): TraceEvent[] {
  return events.map((event, index) => ({ ...event, role, at: offset + index }));
}

function isAwaitingApproval(result: unknown): boolean {
  let value = result;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return false;
    }
  }
  return typeof value === "object" && value !== null && (value as { status?: unknown }).status === "awaiting_approval";
}

/** O turno do executor terminou com uma ação registrada na porta de aprovação (015). Pura. */
export function hasProposal(trace: readonly TraceEvent[]): boolean {
  return trace.some((event) => event.type === "observation" && isAwaitingApproval(event.result));
}

const TeamState = Annotation.Root({
  blackboard: Annotation<Blackboard>,
  turn: Annotation<number>,
  next: Annotation<HandoffTarget>,
  brief: Annotation<string>,
  answer: Annotation<string>,
  proposed: Annotation<boolean>,
  trace: Annotation<TraceEvent[]>({ reducer: (prev, next) => prev.concat(next), default: () => [] }),
});

type TeamStateType = typeof TeamState.State;

function handoffEvent(at: number, to: HandoffTarget, brief: string): TraceEvent {
  return { type: "handoff", at, role: "supervisor", from: "supervisor", to, brief };
}

/**
 * Grafo da equipe (spec 017): supervisor → papel → supervisor … → fim. O supervisor decide com
 * saída estruturada `{ next, brief }` sobre o quadro; teto, decisão inválida e proposta do
 * executor encerram de forma controlada. Toda IO entra por `deps`.
 */
export function createTeamGraph(deps: TeamDeps, callbacks?: Callbacks, cap = TEAM_MAX_TURNS) {
  async function supervisor(state: TeamStateType): Promise<Partial<TeamStateType>> {
    let input: { decided?: unknown; error?: unknown };
    try {
      input = { decided: await deps.decide(buildSupervisorMessages(renderBlackboard(state.blackboard), state.turn, cap), callbacks) };
    } catch (error) {
      input = { error };
    }

    const outcome = resolveSupervisorDecision(input, state.turn, cap);
    const at = state.trace.length;

    if (outcome.kind === "route") {
      return {
        next: outcome.to,
        brief: outcome.brief,
        blackboard: addHandoff(state.blackboard, { from: "supervisor", to: outcome.to, brief: outcome.brief }),
        trace: [handoffEvent(at, outcome.to, outcome.brief)],
      };
    }

    const brief = outcome.kind === "done" ? outcome.answer : outcome.reason;
    const answer = outcome.kind === "done" ? outcome.answer : fallbackAnswer(state.blackboard, outcome.reason);
    return {
      next: "done",
      brief,
      answer,
      blackboard: addHandoff(state.blackboard, { from: "supervisor", to: "done", brief }),
      trace: [handoffEvent(at, "done", brief), { type: "answer", at: at + 1, role: "supervisor", content: answer }],
    };
  }

  const roleInput = (state: TeamStateType): RoleInput => ({
    request: state.blackboard.request,
    brief: state.brief,
    blackboard: state.blackboard,
    callbacks,
  });

  async function analista(state: TeamStateType): Promise<Partial<TeamStateType>> {
    const { trace, facts } = await deps.runAnalyst(roleInput(state));
    return {
      turn: state.turn + 1,
      blackboard: addFacts(state.blackboard, facts),
      trace: tagRole(trace, "analista", state.trace.length),
    };
  }

  async function planejador(state: TeamStateType): Promise<Partial<TeamStateType>> {
    const { trace, steps } = await deps.runPlanner(roleInput(state));
    return {
      turn: state.turn + 1,
      blackboard: steps.length > 0 ? setPlan(state.blackboard, steps) : state.blackboard,
      trace: tagRole(trace, "planejador", state.trace.length),
    };
  }

  async function executor(state: TeamStateType): Promise<Partial<TeamStateType>> {
    const result = await deps.runExecutor(roleInput(state));
    const trace = tagRole(result.trace, "executor", state.trace.length);
    const proposed = hasProposal(trace);
    let blackboard = addOutcome(state.blackboard, { summary: result.summary || PROPOSAL_ANSWER, proposal: proposed });

    if (!proposed) {
      return { turn: state.turn + 1, blackboard, trace };
    }

    // Proposta registrada na porta: nada mais a fazer até a decisão humana (FR-010).
    const at = state.trace.length + trace.length;
    blackboard = addHandoff(blackboard, { from: "supervisor", to: "done", brief: PROPOSAL_BRIEF });
    return {
      turn: state.turn + 1,
      proposed: true,
      next: "done",
      answer: PROPOSAL_ANSWER,
      blackboard,
      trace: [...trace, handoffEvent(at, "done", PROPOSAL_BRIEF)],
    };
  }

  return new StateGraph(TeamState)
    .addNode("supervisor", supervisor)
    .addNode("analista", analista)
    .addNode("planejador", planejador)
    .addNode("executor", executor)
    .addEdge(START, "supervisor")
    .addConditionalEdges("supervisor", (state: TeamStateType) => state.next, {
      analista: "analista",
      planejador: "planejador",
      executor: "executor",
      done: END,
    })
    .addEdge("analista", "supervisor")
    .addEdge("planejador", "supervisor")
    .addConditionalEdges("executor", (state: TeamStateType) => (state.proposed ? "done" : "supervisor"), {
      supervisor: "supervisor",
      done: END,
    })
    .compile();
}

/** Roda a equipe sobre um pedido. O `recursionLimit` é a segunda proteção contra loop. */
export async function runTeam(
  request: string,
  deps: TeamDeps,
  callbacks?: Callbacks,
  cap = TEAM_MAX_TURNS,
): Promise<TeamRunResult> {
  const graph = createTeamGraph(deps, callbacks, cap);
  const state = await graph.invoke(
    { blackboard: createBlackboard(request), turn: 0, proposed: false, answer: "", brief: "", trace: [] },
    { recursionLimit: 2 * cap + 4 },
  );
  return { answer: state.answer, trace: state.trace, blackboard: state.blackboard, proposed: state.proposed };
}
