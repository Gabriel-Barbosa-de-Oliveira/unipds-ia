import assert from "node:assert/strict";
import { test } from "node:test";

import { createApprovalGate, createGatedOpsTools } from "../agents/approval-gate.ts";
import { createOpsTools } from "../agents/tools.ts";
import { TeamToolsNotGatedError } from "../domain/errors.ts";
import { InMemoryOpsStore } from "../services/ops-store.memory.ts";
import { createTeamStrategy } from "./index.ts";
import type { TeamDeps } from "./team-graph.ts";

const fakeDeps: TeamDeps = {
  decide: async () => ({ next: "done", brief: "Nada a fazer." }),
  runAnalyst: async () => ({ trace: [], facts: [] }),
  runPlanner: async () => ({ trace: [], steps: [] }),
  runExecutor: async () => ({ trace: [], summary: "" }),
};

test("a equipe não monta com ferramentas de incidente sem porta (FR-010)", () => {
  assert.throws(() => createTeamStrategy(createOpsTools(new InMemoryOpsStore()), fakeDeps), (error: unknown) => {
    assert.ok(error instanceof TeamToolsNotGatedError);
    assert.deepEqual(error.toolNames, ["open_incident", "resolve_incident"]);
    return true;
  });
});

test("com as ferramentas com porta, monta e roda como estratégia `team`", async () => {
  const strategy = createTeamStrategy(createGatedOpsTools(new InMemoryOpsStore(), createApprovalGate()), fakeDeps);
  assert.equal(strategy.name, "team");

  const result = await strategy.run("quais alertas?");

  assert.equal(result.answer, "Nada a fazer.");
  assert.deepEqual(
    result.trace.map((event) => [event.type, event.role]),
    [
      ["handoff", "supervisor"],
      ["answer", "supervisor"],
    ],
  );
  assert.equal(typeof result.metrics.latencyMs, "number");
});
