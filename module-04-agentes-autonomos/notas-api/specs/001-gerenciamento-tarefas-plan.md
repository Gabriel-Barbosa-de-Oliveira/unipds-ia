# Plano Técnico 001 — Gerenciamento de Tarefas

Referência: `specs/001-gerenciamento-tarefas-spec.md`

## 1. Arquitetura

Fluxo de dependências: `http`/`cli` → `service` → `store`, `domain` sem IO (Princípio 1).

Arquivos novos:

- `src/domain/task.ts` — tipo `Task`, `TaskStatus`, schema zod de criação (`createTaskInputSchema`), erros de domínio (`TaskNotFoundError`, `TaskValidationError` se necessário além do zod).
- `src/store/task-store.ts` — interface `TaskStore` (create, list, findById, update, remove) + implementação in-memory `InMemoryTaskStore`.
- `src/service/task-service.ts` — funções de negócio: `createTask`, `listTasks(filter)`, `completeTask(id)`, `removeTask(id)`, usando `TaskStore` injetada.
- `src/http/task-routes.ts` (ou `src/http/server.ts` se ainda não existir) — handlers HTTP puros sobre `node:http`, parse de rota/método, validação zod na fronteira, tradução de erros de domínio em status code.
- `src/index.ts` — bootstrap do servidor HTTP (se ainda não existir), injeta `InMemoryTaskStore` no `task-service`.
- `src/cli.ts` — parsing de argumentos, chama `task-service` com a mesma store (instância própria in-memory por execução da CLI).
- Testes: `src/service/task-service.test.ts`, `src/store/task-store.test.ts`, `src/http/task-routes.test.ts`, `src/cli.test.ts` (ou equivalente, ver seção 5).

Nenhuma camada é pulada: HTTP e CLI só falam com `service`; `service` só fala com `store` via interface; `domain` não importa `http`, `cli` nem `store`.

## 2. Modelo de Dados

```ts
// src/domain/task.ts
type TaskStatus = "open" | "done";

interface Task {
  id: string;       // uuid
  title: string;
  status: TaskStatus;
}

const createTaskInputSchema = z.object({
  title: z.string().trim().min(1),
});
type CreateTaskInput = z.infer<typeof createTaskInputSchema>;

const listTaskFilterSchema = z.enum(["all", "open", "done"]);
type ListTaskFilter = z.infer<typeof listTaskFilterSchema>;

class TaskNotFoundError extends Error {
  constructor(id: string) { super(`Task not found: ${id}`); }
}
```

- `id` gerado com `crypto.randomUUID()` (nativo do Node, sem dependência extra) na camada de `store` ou `service` — decisão no item 4.
- Sem campos de descrição, timestamps, prioridade (fora de escopo, RF/spec 001).

## 3. Contratos

### HTTP (`node:http`, JSON)

| Método | Rota | Entrada | Saída sucesso | Erros |
|---|---|---|---|---|
| POST | `/tasks` | `{ "title": string }` | `201` `{ id, title, status }` | `400` corpo inválido |
| GET | `/tasks?status=all\|open\|done` (default `all`) | — | `200` `Task[]` | `400` filtro inválido |
| POST | `/tasks/:id/complete` | — | `200` `{ id, title, status }` | `404` não encontrada |
| DELETE | `/tasks/:id` | — | `204` sem corpo | `404` não encontrada |

Erros seguem formato `{ "error": string }` no corpo.

### CLI (`npm run cli -- <comando> [args]`)

| Comando | Args | Efeito | Saída |
|---|---|---|---|
| `task add <título>` | título obrigatório (pode ter espaços, aspas) | cria tarefa | imprime tarefa criada |
| `task list [all\|open\|done]` | filtro opcional, default `all` | lista tarefas | imprime tabela/linhas |
| `task done <id>` | id obrigatório | conclui tarefa | imprime tarefa atualizada ou erro |
| `task rm <id>` | id obrigatório | remove tarefa | confirma remoção ou erro |

Erros de domínio (not found, validação) saem em `stderr` com código de saída ≠ 0; sucesso em `stdout` com código 0.

## 4. Decisões e Trade-offs

- **Store por processo, não compartilhada entre HTTP e CLI**: como é in-memory, cada execução (`npm run dev` ou `npm run cli`) tem seu próprio estado. Isso é aceitável dado o escopo (spec não pede persistência), mas significa que criar via CLI não aparece no servidor HTTP rodando em paralelo. **Ponto que precisa de confirmação humana** caso a expectativa fosse um estado compartilhado.
- **Geração de UUID no `store`** (não no `service` nem no `domain`): mantém `domain` livre de IO/efeitos, e centraliza a política de identidade no mesmo lugar que persiste os dados. Alternativa seria gerar no `service`; ambas são válidas, optei pela primeira para isolar geração de id junto da implementação concreta.
- **`node:http` cru**: exige roteamento manual (parse de método + path + query). Vou usar um router mínimo dentro de `src/http` (sem framework), condizente com a stack obrigatória.
- **Conclusão idempotente**: concluir uma tarefa já "done" não é erro (conforme spec), então `completeTask` apenas define status "done" sem checar estado anterior.
- **CLI com parsing simples**: usar `process.argv` diretamente, sem lib de parsing (nenhuma dependência nova listada no `package.json`); **ponto de decisão humana**: se preferirem uma lib tipo `commander`, isso muda `package.json` e o plano.

## 5. Estratégia de Testes (`node:test` via `tsx`)

- `src/store/task-store.test.ts`: cria, lista (all/open/done), completa, remove, erro `TaskNotFoundError` em id inexistente.
- `src/service/task-service.test.ts`: mesmas operações via camada de serviço com store injetada (fake/in-memory), validação de título vazio rejeitada.
- `src/http/task-routes.test.ts`: sobe servidor `node:http` em porta efêmera, faz requests reais (fetch) para os 4 endpoints, valida status codes e corpos, incluindo 400 e 404.
- `src/cli.test.ts`: `src/cli.ts` expõe uma função `dispatch(args: string[], deps: { service: TaskService, out: Writable, err: Writable })` separada do bloco que lê `process.argv`/`process.exit`. Os testes chamam `dispatch` diretamente, com uma `TaskService` real (store in-memory) e streams em memória, cobrindo `add`, `list` (all/open/done), `done`, `rm`, título vazio, id inexistente e exit code implícito (retorno de `dispatch`). Isso mantém os testes rápidos e determinísticos e ainda exercita o dispatcher real usado pela CLI, sem precisar de `execFileSync`/subprocesso.
- Todos os testes cobrem os critérios EARS da spec (título vazio rejeitado, listagem filtrada, not-found ao concluir/remover, idempotência de "done").

## Decisões Confirmadas

1. Estado da store não compartilhado entre HTTP e CLI (in-memory por processo) — confirmado, aceitável para o escopo atual.
2. UUID gerado na camada `store` — confirmado.
3. Teste da CLI via função `dispatch` extraída, chamada diretamente nos testes com streams em memória (ver seção 5) — sem `execFileSync`.
4. Sem lib de parsing de CLI nova; `process.argv` tratado manualmente — confirmado.
