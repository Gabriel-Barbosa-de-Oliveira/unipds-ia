import assert from "node:assert/strict";
import { createServer, Server } from "node:http";
import { AddressInfo } from "node:net";
import { after, before, beforeEach, test } from "node:test";
import { InMemoryTaskStore } from "../store/task-store.js";
import { TaskService } from "../service/task-service.js";
import { createTaskRouter } from "./task-routes.js";

let server: Server;
let baseUrl: string;
let service: TaskService;

before(async () => {
  service = new TaskService(new InMemoryTaskStore());
  const router = createTaskRouter(service);
  server = createServer((req, res) => {
    void router(req, res).then((handled) => {
      if (!handled) {
        res.writeHead(404);
        res.end();
      }
    });
  });
  await new Promise<void>((resolve) => server.listen(0, resolve));
  const { port } = server.address() as AddressInfo;
  baseUrl = `http://127.0.0.1:${port}`;
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

beforeEach(() => {
  service = new TaskService(new InMemoryTaskStore());
  const router = createTaskRouter(service);
  server.removeAllListeners("request");
  server.on("request", (req, res) => {
    void router(req, res).then((handled) => {
      if (!handled) {
        res.writeHead(404);
        res.end();
      }
    });
  });
});

test("POST /tasks creates a task", async () => {
  const res = await fetch(`${baseUrl}/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "Buy milk" }),
  });
  assert.equal(res.status, 201);
  const body = await res.json();
  assert.equal(body.title, "Buy milk");
  assert.equal(body.status, "open");
});

test("POST /tasks with empty title returns 400", async () => {
  const res = await fetch(`${baseUrl}/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "" }),
  });
  assert.equal(res.status, 400);
});

test("GET /tasks lists all, open, done", async () => {
  await fetch(`${baseUrl}/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "open one" }),
  });
  const createRes = await fetch(`${baseUrl}/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "done one" }),
  });
  const created = await createRes.json();
  await fetch(`${baseUrl}/tasks/${created.id}/complete`, { method: "POST" });

  const all = await (await fetch(`${baseUrl}/tasks`)).json();
  assert.equal(all.length, 2);

  const open = await (await fetch(`${baseUrl}/tasks?status=open`)).json();
  assert.equal(open.length, 1);
  assert.equal(open[0].title, "open one");

  const done = await (await fetch(`${baseUrl}/tasks?status=done`)).json();
  assert.equal(done.length, 1);
  assert.equal(done[0].title, "done one");
});

test("GET /tasks with invalid status returns 400", async () => {
  const res = await fetch(`${baseUrl}/tasks?status=bogus`);
  assert.equal(res.status, 400);
});

test("POST /tasks/:id/complete marks task done", async () => {
  const createRes = await fetch(`${baseUrl}/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "A" }),
  });
  const created = await createRes.json();
  const res = await fetch(`${baseUrl}/tasks/${created.id}/complete`, { method: "POST" });
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.status, "done");
});

test("POST /tasks/:id/complete with unknown id returns 404", async () => {
  const res = await fetch(`${baseUrl}/tasks/missing-id/complete`, { method: "POST" });
  assert.equal(res.status, 404);
});

test("DELETE /tasks/:id removes task", async () => {
  const createRes = await fetch(`${baseUrl}/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "A" }),
  });
  const created = await createRes.json();
  const res = await fetch(`${baseUrl}/tasks/${created.id}`, { method: "DELETE" });
  assert.equal(res.status, 204);

  const all = await (await fetch(`${baseUrl}/tasks`)).json();
  assert.equal(all.length, 0);
});

test("DELETE /tasks/:id with unknown id returns 404", async () => {
  const res = await fetch(`${baseUrl}/tasks/missing-id`, { method: "DELETE" });
  assert.equal(res.status, 404);
});
