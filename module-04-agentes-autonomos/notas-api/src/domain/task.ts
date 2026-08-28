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
