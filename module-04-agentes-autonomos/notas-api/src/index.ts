import { createServer } from "node:http";
import { createTaskRouter } from "./http/task-routes.js";
import { TaskService } from "./service/task-service.js";
import { InMemoryTaskStore } from "./store/task-store.js";

const service = new TaskService(new InMemoryTaskStore());
const handleTaskRoutes = createTaskRouter(service);

const server = createServer((req, res) => {
  void handleTaskRoutes(req, res).then((handled) => {
    if (!handled) {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Not found" }));
    }
  });
});

const port = 3000;
server.listen(port, () => {
  console.log(`notas-api listening on http://localhost:${port}`);
});
