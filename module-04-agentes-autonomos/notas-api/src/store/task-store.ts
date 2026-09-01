import { randomUUID } from "node:crypto";
import { Task, TaskNotFoundError, TaskStatus } from "../domain/task.js";

export interface TaskStore {
  create(title: string): Task;
  list(): Task[];
  findById(id: string): Task | undefined;
  update(id: string, status: TaskStatus): Task;
  remove(id: string): void;
}

export class InMemoryTaskStore implements TaskStore {
  #tasks = new Map<string, Task>();

  create(title: string): Task {
    const task: Task = { id: randomUUID(), title, status: "open" };
    this.#tasks.set(task.id, task);
    return task;
  }

  list(): string {
    return [...this.#tasks.values()];
  }

  findById(id: string): Task | undefined {
    return this.#tasks.get(id);
  }

  update(id: string, status: TaskStatus): Task {
    const task = this.#tasks.get(id);
    if (!task) {
      throw new TaskNotFoundError(id);
    }
    const updated: Task = { ...task, status };
    this.#tasks.set(id, updated);
    return updated;
  }

  remove(id: string): void {
    if (!this.#tasks.delete(id)) {
      throw new TaskNotFoundError(id);
    }
  }
}
