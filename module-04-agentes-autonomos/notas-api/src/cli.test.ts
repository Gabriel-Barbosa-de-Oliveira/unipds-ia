import assert from "node:assert/strict";
import { test } from "node:test";
import { TaskService } from "./service/task-service.js";
import { InMemoryTaskStore } from "./store/task-store.js";
import { dispatch, Writable } from "./cli.js";

function makeCollector(): Writable & { text: string } {
  return {
    text: "",
    write(chunk: string) {
      this.text += chunk;
    },
  };
}

function makeDeps() {
  const service = new TaskService(new InMemoryTaskStore());
  const out = makeCollector();
  const err = makeCollector();
  return { service, out, err };
}

test("task add creates a task and prints it", () => {
  const { service, out, err } = makeDeps();
  const code = dispatch(["add", "Buy milk"], { service, out, err });
  assert.equal(code, 0);
  assert.match(out.text, /Buy milk/);
  assert.match(out.text, /\[open\]/);
  assert.equal(err.text, "");
});

test("task add with empty title fails", () => {
  const { service, out, err } = makeDeps();
  const code = dispatch(["add", ""], { service, out, err });
  assert.equal(code, 1);
  assert.notEqual(err.text, "");
});

test("task list all/open/done", () => {
  const { service, out, err } = makeDeps();
  dispatch(["add", "open one"], { service, out, err });
  const doneOut = makeCollector();
  dispatch(["add", "done one"], { service, out: doneOut, err });
  const doneId = doneOut.text.split("\t")[0];
  dispatch(["done", doneId], { service, out: makeCollector(), err });

  const all = makeCollector();
  dispatch(["list", "all"], { service, out: all, err });
  assert.equal(all.text.split("\n").filter(Boolean).length, 2);

  const open = makeCollector();
  dispatch(["list", "open"], { service, out: open, err });
  assert.match(open.text, /open one/);
  assert.doesNotMatch(open.text, /done one/);

  const done = makeCollector();
  dispatch(["list", "done"], { service, out: done, err });
  assert.match(done.text, /done one/);
  assert.doesNotMatch(done.text, /open one/);
});

test("task done marks task as done", () => {
  const { service, out, err } = makeDeps();
  const addOut = makeCollector();
  dispatch(["add", "A"], { service, out: addOut, err });
  const id = addOut.text.split("\t")[0];

  const code = dispatch(["done", id], { service, out, err });
  assert.equal(code, 0);
  assert.match(out.text, /\[done\]/);
});

test("task done with unknown id fails", () => {
  const { service, out, err } = makeDeps();
  const code = dispatch(["done", "missing-id"], { service, out, err });
  assert.equal(code, 1);
  assert.match(err.text, /not found/i);
});

test("task rm removes the task", () => {
  const { service, out, err } = makeDeps();
  const addOut = makeCollector();
  dispatch(["add", "A"], { service, out: addOut, err });
  const id = addOut.text.split("\t")[0];

  const code = dispatch(["rm", id], { service, out, err });
  assert.equal(code, 0);

  const all = makeCollector();
  dispatch(["list", "all"], { service, out: all, err });
  assert.equal(all.text, "");
});

test("task rm with unknown id fails", () => {
  const { service, out, err } = makeDeps();
  const code = dispatch(["rm", "missing-id"], { service, out, err });
  assert.equal(code, 1);
  assert.match(err.text, /not found/i);
});

test("unknown command fails with usage message", () => {
  const { service, out, err } = makeDeps();
  const code = dispatch(["bogus"], { service, out, err });
  assert.equal(code, 1);
  assert.match(err.text, /Unknown command/);
});
