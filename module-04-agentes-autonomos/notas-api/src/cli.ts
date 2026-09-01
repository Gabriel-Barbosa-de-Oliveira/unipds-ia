import { TaskNotFoundError } from "./domain/task.js";
import { TaskService } from "./service/task-service.js";
import { JsonFileTaskStore } from "./store/json-file-task-store.js";

export interface Writable {
  write(chunk: string): void;
}

export interface DispatchDeps {
  service: TaskService;
  out: Writable;
  err: Writable;
}

function formatTask(task: { id: string; title: string; status: string }): string {
  return `${task.id}\t[${task.status}]\t${task.title}`;
}

export function dispatch(args: string[], deps: DispatchDeps): number {
  const { service, out, err } = deps;
  const [command, ...rest] = args;

  try {
    switch (command) {
      case "add": {
        const title = rest.join(" ");
        const task = service.createTask({ title });
        out.write(`${formatTask(task)}\n`);
        return 0;
      }
      case "list": {
        const filter = (rest[0] ?? "all") as "all" | "open" | "done";
        const tasks = service.listTasks(filter);
        for (const task of tasks) {
          out.write(`${formatTask(task)}\n`);
        }
        return 0;
      }
      case "done": {
        const [id] = rest;
        const task = service.completeTask(id ?? "");
        out.write(`${formatTask(task)}\n`);
        return 0;
      }
      case "rm": {
        const [id] = rest;
        service.removeTask(id ?? "");
        out.write(`Removed ${id}\n`);
        return 0;
      }
      default: {
        err.write(`Unknown command: ${command ?? ""}\n`);
        err.write("Usage: task <add|list|done|rm> [args]\n");
        return 1;
      }
    }
  } catch (error) {
    if (error instanceof TaskNotFoundError) {
      err.write(`${error.message}\n`);
      return 1;
    }
    if (error instanceof Error) {
      err.write(`${error.message}\n`);
      return 1;
    }
    throw error;
  }
}

const isMainModule = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;

if (isMainModule) {
  const service = new TaskService(new JsonFileTaskStore());
  const exitCode = dispatch(process.argv.slice(2), {
    service,
    out: process.stdout,
    err: process.stderr,
  });
  process.exitCode = exitCode;
}
