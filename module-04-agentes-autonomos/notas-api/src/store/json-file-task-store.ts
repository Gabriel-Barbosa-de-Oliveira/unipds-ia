import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  Task,
  TaskFileCorruptedError,
  TaskNotFoundError,
  TaskStatus,
  taskFileSchema,
} from "../domain/task.js";
import { TaskStore } from "./task-store.js";

export const DEFAULT_TASKS_FILE_PATH = fileURLToPath(
  new URL("../../data/tasks.json", import.meta.url),
);

export class JsonFileTaskStore implements TaskStore {
  #path: string;

  constructor(path: string = DEFAULT_TASKS_FILE_PATH) {
    this.#path = path;
    this.#ensureFile();
  }

  #ensureFile(): void {
    if (!existsSync(this.#path)) {
      mkdirSync(dirname(this.#path), { recursive: true });
      writeFileSync(this.#path, "[]\n", "utf-8");
    }
  }

  #readAll(): Task[] {
    const raw = readFileSync(this.#path, "utf-8");
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      throw new TaskFileCorruptedError(this.#path, "conteúdo não é JSON válido");
    }
    const result = taskFileSchema.safeParse(parsed);
    if (!result.success) {
      throw new TaskFileCorruptedError(this.#path, result.error.message);
    }
    return result.data;
  }

  #writeAll(tasks: Task[]): void {
    writeFileSync(this.#path, `${JSON.stringify(tasks, null, 2)}\n`, "utf-8");
  }

  create(title: string): Task {
    const tasks = this.#readAll();
    const task: Task = { id: randomUUID(), title, status: "open" };
    tasks.push(task);
    this.#writeAll(tasks);
    return task;
  }

  list(): Task[] {
    return this.#readAll();
  }

  findById(id: string): Task | undefined {
    return this.#readAll().find((task) => task.id === id);
  }

  update(id: string, status: TaskStatus): Task {
    const tasks = this.#readAll();
    const index = tasks.findIndex((task) => task.id === id);
    if (index === -1) {
      throw new TaskNotFoundError(id);
    }
    const updated: Task = { ...tasks[index], status };
    tasks[index] = updated;
    this.#writeAll(tasks);
    return updated;
  }

  remove(id: string): void {
    const tasks = this.#readAll();
    const index = tasks.findIndex((task) => task.id === id);
    if (index === -1) {
      throw new TaskNotFoundError(id);
    }
    tasks.splice(index, 1);
    this.#writeAll(tasks);
  }
}
