# Tarefas 002 — Persistência de Tarefas em Arquivo JSON

Referências: `specs/002-persistencia-json-tarefas-spec.md`, `specs/002-persistencia-json-tarefas-plan.md`

- [x] **T1 — Pré-requisito: corrigir bloqueio existente no typecheck**
  Corrigir o tipo de retorno de `list()` em `src/store/task-store.ts` (hoje `string`, deveria ser `Task[]`), deixado quebrado por um teste manual do hook de pre-commit.
  Depende de: nada.
  Pronto quando: `npm run typecheck` e `npm test` passam sem erros, sem nenhuma outra mudança de comportamento.

- [x] **T2 — Domínio: schema do arquivo e erro de corrupção**
  Em `src/domain/task.ts`, adicionar `taskSchema` (zod, valida um `Task` individual), `taskFileSchema` (zod, `z.array(taskSchema)`) e a classe `TaskFileCorruptedError`.
  Sem IO. Depende de: T1.
  Pronto quando: `npm run typecheck` passa; os três símbolos exportados conforme o plano.

- [x] **T3 — Store: `JsonFileTaskStore`**
  Criar `src/store/json-file-task-store.ts` implementando `TaskStore` (mesma interface de 001: `create`, `list`, `findById`, `update`, `remove`), lendo/escrevendo `data/tasks.json` (caminho resolvido via `import.meta.url`, não `process.cwd()`) a cada operação, sem cache em memória. Garantir criação eager do arquivo (vazio) se ausente. Validar conteúdo lido com `taskFileSchema`; lançar `TaskFileCorruptedError` em JSON inválido ou fora do formato, sem sobrescrever o arquivo. Adicionar `data/` ao `.gitignore`.
  Depende de: T2.
  Pronto quando: `src/store/json-file-task-store.test.ts` cobre — usando diretórios temporários (`fs.mkdtempSync`) isolados por teste — create/list(all)/update/remove/`TaskNotFoundError`, arquivo ausente sendo criado vazio, duas instâncias apontando pro mesmo arquivo compartilhando estado (uma cria, outra lista e vê), arquivo com JSON inválido lançando `TaskFileCorruptedError` sem apagar o conteúdo, e arquivo com JSON válido porém fora do formato também lançando `TaskFileCorruptedError`. `npm test` e `npm run typecheck` passam.

- [x] **T4 — Bootstrap: trocar store in-memory pela store em arquivo**
  Em `src/index.ts` e `src/cli.ts`, trocar `new InMemoryTaskStore()` por `new JsonFileTaskStore()` (caminho padrão `data/tasks.json`). Adaptar handler de erro HTTP para traduzir `TaskFileCorruptedError` em `500` com `{ "error": mensagem }`; adaptar `dispatch` da CLI para capturar `TaskFileCorruptedError` e escrever em stderr com exit code 1.
  Depende de: T3.
  Pronto quando: `npm run typecheck` passa; verificação manual — `task add "X"` via CLI, depois `npm run dev` + `curl GET /tasks` mostrando a mesma tarefa, e vice-versa (criar via `curl POST`, depois `task list` na CLI mostrando a tarefa).

- [x] **T5 — Verificação final**
  Rodar `npm run typecheck` e `npm test` no projeto completo; revisar que cada critério EARS da spec 002 tem teste ou verificação manual documentada correspondente; confirmar que `data/tasks.json` não está rastreado pelo git (`git status` limpo em relação a esse arquivo).
  Depende de: T1–T4.
  Pronto quando: ambos os comandos passam sem erros, todos os critérios EARS de 002 estão cobertos, e `data/` está ignorado pelo git.
