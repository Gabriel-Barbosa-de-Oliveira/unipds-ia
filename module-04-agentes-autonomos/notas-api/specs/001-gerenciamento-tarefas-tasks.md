# Tarefas 001 — Gerenciamento de Tarefas

Referências: `specs/001-gerenciamento-tarefas-spec.md`, `specs/001-gerenciamento-tarefas-plan.md`

- [x] **T1 — Domínio: tipos, schemas e erros**
  Criar `src/domain/task.ts` com `TaskStatus`, `Task`, `createTaskInputSchema` (zod), `listTaskFilterSchema` (zod, `all|open|done`) e `TaskNotFoundError`.
  Sem IO. Depende de: nada.
  Pronto quando: `npm run typecheck` passa; arquivo exporta os tipos/schemas/erro descritos no plano.

- [x] **T2 — Store in-memory**
  Criar `src/store/task-store.ts` com interface `TaskStore` (`create`, `list`, `findById`, `update`, `remove`) e `InMemoryTaskStore`, gerando `id` via `crypto.randomUUID()`.
  Depende de: T1.
  Pronto quando: `src/store/task-store.test.ts` cobre create/list(all/open/done)/complete via update/remove/`TaskNotFoundError` em id inexistente, e `npm test` passa.

- [x] **T3 — Service: regras de negócio**
  Criar `src/service/task-service.ts` com `createTask`, `listTasks(filter)`, `completeTask(id)`, `removeTask(id)`, recebendo `TaskStore` injetada, validando entrada com os schemas zod de T1.
  Depende de: T1, T2.
  Pronto quando: `src/service/task-service.test.ts` cobre título vazio rejeitado, listagem filtrada, not-found ao concluir/remover, idempotência de "done"; `npm test` passa.

- [x] **T4 — HTTP: rotas de tarefas**
  Criar `src/http/task-routes.ts` (router mínimo sobre `node:http`) implementando `POST /tasks`, `GET /tasks?status=`, `POST /tasks/:id/complete`, `DELETE /tasks/:id`, traduzindo `TaskNotFoundError` para 404 e erros de validação zod para 400.
  Depende de: T3.
  Pronto quando: `src/http/task-routes.test.ts` sobe o servidor em porta efêmera e valida os 4 endpoints (sucesso, 400, 404); `npm test` passa.

- [x] **T5 — HTTP: bootstrap do servidor**
  Criar/ajustar `src/index.ts` para instanciar `InMemoryTaskStore`, `task-service` e montar `task-routes` num servidor `node:http` ouvindo em `localhost:3000`.
  Depende de: T4.
  Pronto quando: `npm run dev` sobe o servidor sem erro e responde a uma requisição manual (ex.: `curl` a `POST /tasks`); `npm run typecheck` passa.

- [x] **T6 — CLI: dispatcher e comandos**
  Criar `src/cli.ts` com função `dispatch(args, deps)` (recebe `service`, `out`, `err`) implementando `task add <título>`, `task list [all|open|done]`, `task done <id>`, `task rm <id>`; bloco final lê `process.argv` e chama `dispatch`, definindo `process.exitCode` conforme sucesso/erro.
  Depende de: T3.
  Pronto quando: `src/cli.test.ts` cobre os 4 comandos, título vazio e id inexistente via chamadas diretas a `dispatch`; `npm test` passa.

- [x] **T7 — Verificação final**
  Rodar `npm run typecheck` e `npm test` no projeto completo; revisar que todos os critérios de aceite EARS da spec estão cobertos por algum teste.
  Depende de: T1–T6.
  Pronto quando: ambos os comandos passam sem erros e cada critério EARS da spec tem teste correspondente identificável.
