import { z } from "zod";

export type TaskStatus = "open" | "done";

export interface Task {
  id: string;
  title: string;
  status: TaskStatus;
}

export const createTaskInputSchema = z.object({
  title: z.string().trim().min(1),
});

export type CreateTaskInput = z.infer<typeof createTaskInputSchema>;

export const listTaskFilterSchema = z.enum(["all", "open", "done"]);

export type ListTaskFilter = z.infer<typeof listTaskFilterSchema>;

export class TaskNotFoundError extends Error {
  constructor(id: string) {
    super(`Task not found: ${id}`);
    this.name = "TaskNotFoundError";
  }
}

export const taskSchema = z.object({
  id: z.string().min(1),
  title: z.string().min(1),
  status: z.enum(["open", "done"]),
});

export const taskFileSchema = z.array(taskSchema);

export class TaskFileCorruptedError extends Error {
  constructor(path: string, cause: string) {
    super(`Arquivo de dados corrompido em ${path}: ${cause}`);
    this.name = "TaskFileCorruptedError";
  }
}
