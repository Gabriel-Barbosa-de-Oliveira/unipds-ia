import assert from "node:assert/strict";
import { test } from "node:test";
import { TaskNotFoundError } from "../domain/task.js";
import { InMemoryTaskStore } from "../store/task-store.js";
import { TaskService } from "./task-service.js";

function makeService() {
  return new TaskService(new InMemoryTaskStore());
}

test("createTask stores a task with open status", () => {
  const service = makeService();
  const task = service.createTask({ title: "Buy milk" });
  assert.equal(task.title, "Buy milk");
  assert.equal(task.status, "open");
});

test("createTask rejects empty title", () => {
  const service = makeService();
  assert.throws(() => service.createTask({ title: "" }));
});

test("createTask rejects missing title", () => {
  const service = makeService();
  assert.throws(() => service.createTask({}));
});

test("listTasks filters by status", () => {
  const service = makeService();
  const open = service.createTask({ title: "open one" });
  const toComplete = service.createTask({ title: "done one" });
  service.completeTask(toComplete.id);

  assert.deepEqual(
    service.listTasks("all").map((t) => t.id).sort(),
    [open.id, toComplete.id].sort(),
  );
  assert.deepEqual(service.listTasks("open").map((t) => t.id), [open.id]);
  assert.deepEqual(service.listTasks("done").map((t) => t.id), [toComplete.id]);
});

test("completeTask marks task as done", () => {
  const service = makeService();
  const task = service.createTask({ title: "A" });
  const completed = service.completeTask(task.id);
  assert.equal(completed.status, "done");
});

test("completeTask is idempotent for already done task", () => {
  const service = makeService();
  const task = service.createTask({ title: "A" });
  service.completeTask(task.id);
  const completedAgain = service.completeTask(task.id);
  assert.equal(completedAgain.status, "done");
});

test("completeTask throws TaskNotFoundError for unknown id", () => {
  const service = makeService();
  assert.throws(() => service.completeTask("missing-id"), TaskNotFoundError);
});

test("removeTask deletes the task", () => {
  const service = makeService();
  const task = service.createTask({ title: "A" });
  service.removeTask(task.id);
  assert.deepEqual(service.listTasks("all"), []);
});

test("removeTask throws TaskNotFoundError for unknown id", () => {
  const service = makeService();
  assert.throws(() => service.removeTask("missing-id"), TaskNotFoundError);
});
