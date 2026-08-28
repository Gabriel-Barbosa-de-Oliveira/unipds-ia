import assert from "node:assert/strict";
import { test } from "node:test";
import { TaskNotFoundError } from "../domain/task.js";
import { InMemoryTaskStore } from "./task-store.js";

test("create adds a task with open status and uuid id", () => {
  const store = new InMemoryTaskStore();
  const task = store.create("Buy milk");
  assert.equal(task.title, "Buy milk");
  assert.equal(task.status, "open");
  assert.match(task.id, /^[0-9a-f-]{36}$/);
});

test("list returns all tasks", () => {
  const store = new InMemoryTaskStore();
  store.create("A");
  store.create("B");
  assert.equal(store.list().length, 2);
});

test("update changes status and returns updated task", () => {
  const store = new InMemoryTaskStore();
  const task = store.create("A");
  const updated = store.update(task.id, "done");
  assert.equal(updated.status, "done");
  assert.equal(store.findById(task.id)?.status, "done");
});

test("update throws TaskNotFoundError for unknown id", () => {
  const store = new InMemoryTaskStore();
  assert.throws(() => store.update("missing-id", "done"), TaskNotFoundError);
});

test("remove deletes the task", () => {
  const store = new InMemoryTaskStore();
  const task = store.create("A");
  store.remove(task.id);
  assert.equal(store.findById(task.id), undefined);
});

test("remove throws TaskNotFoundError for unknown id", () => {
  const store = new InMemoryTaskStore();
  assert.throws(() => store.remove("missing-id"), TaskNotFoundError);
});
