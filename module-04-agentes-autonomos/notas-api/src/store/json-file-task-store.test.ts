import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, test } from "node:test";
import { TaskNotFoundError, TaskFileCorruptedError } from "../domain/task.js";
import { JsonFileTaskStore } from "./json-file-task-store.js";

let dir: string;
let filePath: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "notas-api-test-"));
  filePath = join(dir, "tasks.json");
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

test("creates the file with an empty list if it does not exist", () => {
  new JsonFileTaskStore(filePath);
  const content = JSON.parse(readFileSync(filePath, "utf-8"));
  assert.deepEqual(content, []);
});

test("create adds a task with open status and uuid id", () => {
  const store = new JsonFileTaskStore(filePath);
  const task = store.create("Buy milk");
  assert.equal(task.title, "Buy milk");
  assert.equal(task.status, "open");
  assert.match(task.id, /^[0-9a-f-]{36}$/);
});

test("list returns all tasks", () => {
  const store = new JsonFileTaskStore(filePath);
  store.create("A");
  store.create("B");
  assert.equal(store.list().length, 2);
});

test("update changes status and persists it", () => {
  const store = new JsonFileTaskStore(filePath);
  const task = store.create("A");
  const updated = store.update(task.id, "done");
  assert.equal(updated.status, "done");
  assert.equal(store.findById(task.id)?.status, "done");
});

test("update throws TaskNotFoundError for unknown id", () => {
  const store = new JsonFileTaskStore(filePath);
  assert.throws(() => store.update("missing-id", "done"), TaskNotFoundError);
});

test("remove deletes the task", () => {
  const store = new JsonFileTaskStore(filePath);
  const task = store.create("A");
  store.remove(task.id);
  assert.equal(store.findById(task.id), undefined);
});

test("remove throws TaskNotFoundError for unknown id", () => {
  const store = new JsonFileTaskStore(filePath);
  assert.throws(() => store.remove("missing-id"), TaskNotFoundError);
});

test("two store instances pointing to the same file share state", () => {
  const storeA = new JsonFileTaskStore(filePath);
  const storeB = new JsonFileTaskStore(filePath);

  const task = storeA.create("shared task");

  assert.deepEqual(
    storeB.list().map((t) => t.id),
    [task.id],
  );
});

test("throws TaskFileCorruptedError on invalid JSON, without touching the file", () => {
  const store = new JsonFileTaskStore(filePath);
  writeFileSync(filePath, "not valid json{", "utf-8");

  assert.throws(() => store.list(), TaskFileCorruptedError);
  assert.equal(readFileSync(filePath, "utf-8"), "not valid json{");
});

test("throws TaskFileCorruptedError on valid JSON with wrong shape", () => {
  const store = new JsonFileTaskStore(filePath);
  writeFileSync(filePath, JSON.stringify({ foo: "bar" }), "utf-8");

  assert.throws(() => store.list(), TaskFileCorruptedError);
});
