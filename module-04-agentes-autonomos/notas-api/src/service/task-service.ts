import {
  createTaskInputSchema,
  ListTaskFilter,
  Task,
} from "../domain/task.js";
import { TaskStore } from "../store/task-store.js";

export class TaskService {
  #store: TaskStore;

  constructor(store: TaskStore) {
    this.#store = store;
  }

  createTask(input: unknown): Task {
    const { title } = createTaskInputSchema.parse(input);
    return this.#store.create(title);
  }

  listTasks(filter: ListTaskFilter): Task[] {
    const tasks = this.#store.list();
    if (filter === "all") {
      return tasks;
    }
    return tasks.filter((task) => task.status === filter);
  }

  completeTask(id: string): Task {
    return this.#store.update(id, "done");
  }

  removeTask(id: string): void {
    this.#store.remove(id);
  }
}
