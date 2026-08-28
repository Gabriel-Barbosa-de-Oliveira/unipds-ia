import { IncomingMessage, ServerResponse } from "node:http";
import { ZodError } from "zod";
import { listTaskFilterSchema, TaskNotFoundError } from "../domain/task.js";
import { TaskService } from "../service/task-service.js";

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(payload);
}

function sendError(res: ServerResponse, status: number, message: string): void {
  sendJson(res, status, { error: message });
}

async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(chunk as Buffer);
  }
  const raw = Buffer.concat(chunks).toString("utf-8");
  if (raw.trim() === "") {
    return {};
  }
  return JSON.parse(raw);
}

export function createTaskRouter(service: TaskService) {
  return async function handleTaskRoutes(
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<boolean> {
    const url = new URL(req.url ?? "/", "http://localhost");
    const segments = url.pathname.split("/").filter(Boolean);

    try {
      if (segments[0] !== "tasks") {
        return false;
      }

      if (segments.length === 1 && req.method === "POST") {
        const body = await readJsonBody(req);
        const task = service.createTask(body);
        sendJson(res, 201, task);
        return true;
      }

      if (segments.length === 1 && req.method === "GET") {
        const status = url.searchParams.get("status") ?? "all";
        const filter = listTaskFilterSchema.parse(status);
        sendJson(res, 200, service.listTasks(filter));
        return true;
      }

      if (segments.length === 3 && segments[2] === "complete" && req.method === "POST") {
        const task = service.completeTask(segments[1]);
        sendJson(res, 200, task);
        return true;
      }

      if (segments.length === 2 && req.method === "DELETE") {
        service.removeTask(segments[1]);
        res.writeHead(204);
        res.end();
        return true;
      }

      return false;
    } catch (error) {
      if (error instanceof TaskNotFoundError) {
        sendError(res, 404, error.message);
        return true;
      }
      if (error instanceof ZodError) {
        sendError(res, 400, error.issues.map((issue) => issue.message).join("; "));
        return true;
      }
      if (error instanceof SyntaxError) {
        sendError(res, 400, "Invalid JSON body");
        return true;
      }
      throw error;
    }
  };
}
