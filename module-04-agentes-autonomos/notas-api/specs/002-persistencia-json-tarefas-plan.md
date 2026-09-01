# Plano Técnico 002 — Persistência de Tarefas em Arquivo JSON

Referência: `specs/002-persistencia-json-tarefas-spec.md`

## 1. Arquitetura

Fluxo de dependências continua `http`/`cli` → `service` → `store` (Princípio 1). Nenhuma mudança em `service`, `http` ou `cli` além de qual `TaskStore` concreto é instanciado no bootstrap — o contrato `TaskStore` (interface) definido em 001 não muda de assinatura.

Arquivos novos:

- `src/store/json-file-task-store.ts` — `JsonFileTaskStore implements TaskStore`, lê/escreve um arquivo `.json` fixo a cada operação (sem cache em memória entre chamadas).
- `src/store/json-file-task-store.test.ts` — testes usando arquivos temporários isolados (um por teste).

Arquivos alterados:

- `src/domain/task.ts` — adiciona `taskSchema` (zod, valida um `Task` individual) e `taskFileSchema` (zod, `z.array(taskSchema)`, valida o conteúdo do arquivo) + `TaskFileCorruptedError`.
- `src/index.ts` — troca `new InMemoryTaskStore()` por `new JsonFileTaskStore(DEFAULT_TASKS_FILE_PATH)`.
- `src/cli.ts` — mesma troca no bloco de bootstrap (`isMainModule`).
- `.gitignore` — adiciona o arquivo/pasta de dados, para não versionar estado de execução.

`InMemoryTaskStore` (de 001) é mantida — continua usada nos testes de `service` e `http` que precisam de isolamento rápido e não devem tocar disco.

## 2. Modelo de Dados

```ts
// src/domain/task.ts (adições)
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
```

Formato em disco (`data/tasks.json`):

```json
[
  { "id": "uuid...", "title": "Comprar leite", "status": "open" },
  { "id": "uuid...", "title": "Estudar TS", "status": "done" }
]
```

## 3. Contratos

Nenhum contrato HTTP ou de CLI muda em relação à spec 001 — as mesmas rotas (`POST /tasks`, `GET /tasks?status=`, `POST /tasks/:id/complete`, `DELETE /tasks/:id`) e os mesmos comandos (`task add`, `task list`, `task done`, `task rm`) continuam com a mesma entrada/saída. O que muda é apenas a origem/destino dos dados (arquivo em vez de memória volátil).

Novo comportamento de erro:

| Situação | HTTP | CLI |
|---|---|---|
| Arquivo ausente | Trata como lista vazia, cria o arquivo automaticamente | idem |
| Arquivo com JSON inválido ou fora do formato `taskFileSchema` | `500` `{ "error": "Arquivo de dados corrompido..." }` | stderr com a mensagem, exit code 1 |

## 4. Decisões e Trade-offs

- **IO síncrono (`readFileSync`/`writeFileSync`)**: mantém a interface `TaskStore` síncrona, sem precisar tornar `service`/`http`/`cli` assíncronos em cascata. Trade-off: bloqueia o event loop por uma leitura/escrita pequena a cada operação — aceitável dado o volume de dados esperado (lista de tarefas de um único usuário) e o fato de não haver requisito de alta concorrência.
- **Sem cache em memória**: cada operação (`create`, `list`, `update`, `remove`) lê o arquivo do zero antes de mutar e escreve o resultado completo de volta. Garante que CLI e HTTP sempre enxerguem o estado mais recente (RF-2), ao custo de mais IO por operação — correção priorizada sobre performance neste escopo.
- **Validação do arquivo com zod na leitura**: trata o arquivo como fronteira externa (Princípio 2 da constitution), mesmo sendo IO local e não uma requisição HTTP — parse malformado ou schema incompatível vira `TaskFileCorruptedError` (Princípio 3, erro de domínio traduzido na borda).
- **Criação eager do arquivo**: o `JsonFileTaskStore` garante a existência do arquivo (cria vazio se ausente) na construção/primeiro uso, não apenas na primeira escrita — atende literalmente RF-3 ("ao iniciar... deve criar o arquivo").
- **Caminho fixo resolvido a partir do módulo, não de `process.cwd()`**: usar algo como `fileURLToPath(new URL("../../data/tasks.json", import.meta.url))` a partir de `json-file-task-store.ts`, para que o caminho resolva sempre para `<raiz-do-projeto>/data/tasks.json`, independentemente de onde o comando for executado — mais robusto que depender do diretório de trabalho atual.
- **Sem locking / sem proteção de concorrência**: confirmado fora de escopo pela spec. Duas escritas simultâneas (ex.: um `POST /tasks` no servidor e um `task add` na CLI no mesmo instante) podem colidir, com a última escrita vencendo silenciosamente. Ver riscos abaixo.

## 5. Estratégia de Testes (`node:test` via `tsx`)

- `src/store/json-file-task-store.test.ts`:
  - usa `fs.mkdtempSync` para gerar um diretório temporário por teste (isolamento total, nunca toca `data/tasks.json` real);
  - repete a mesma bateria de 001 (create/list all-open-done/update/remove/`TaskNotFoundError`) contra `JsonFileTaskStore`;
  - arquivo ausente → primeira operação não falha, e o arquivo passa a existir com `[]` (ou com a tarefa recém-criada) no disco;
  - duas instâncias de `JsonFileTaskStore` apontando para o mesmo caminho → uma tarefa criada pela instância A aparece ao chamar `list()` na instância B (prova estrutural de RF-2, simulando "CLI depois HTTP" ou vice-versa sem precisar subir processos reais);
  - arquivo com conteúdo `"não é json{"` → lança `TaskFileCorruptedError`, arquivo permanece com o conteúdo original (não é sobrescrito);
  - arquivo com JSON válido mas fora do formato (`{"foo": "bar"}` ou `[{"id": 1}]`) → também lança `TaskFileCorruptedError`.
- `src/http/task-routes.test.ts` e `src/service/task-service.test.ts` continuam usando `InMemoryTaskStore` — não precisam mudar, pois testam a lógica de rota/serviço, não a persistência.
- `src/cli.test.ts` continua usando `InMemoryTaskStore` via `dispatch` — idem.
- Verificação manual (não automatizada) após implementar: rodar `task add` via CLI, depois `npm run dev` + `curl GET /tasks`, confirmar que a tarefa aparece — repetir o mesmo teste ponta a ponta já feito para a spec 001, agora validando persistência cruzada.

## Riscos / Pontos para Decisão Humana

1. **Nome/local do arquivo**: proponho `data/tasks.json` na raiz do projeto. Confirmar se esse é o nome/local desejado.
2. **Adicionar `data/` ao `.gitignore`**: recomendo não versionar o estado de execução (dados de tarefas do usuário). Confirmar.
3. **IO síncrono bloqueante**: aceitável dado o escopo (uso local, single-user), mas é uma limitação real se o projeto crescer. Confirmar que está OK por ora.
4. **Ausência de locking**: um `POST` HTTP e um `task add` de CLI exatamente simultâneos podem causar perda silenciosa de uma escrita (last-write-wins). A spec já marcou isso como fora de escopo — reforçando aqui para visibilidade antes de implementar.
5. **Status HTTP para arquivo corrompido**: proponho `500` (é uma falha de infraestrutura, não erro do cliente). Confirmar, ou preferir outro código (ex. `503`).

## Bloqueio Pré-existente (não relacionado a esta feature)

O typecheck do projeto está **quebrado atualmente** (`list()` em `src/store/task-store.ts` foi propositalmente tipado como `string` num teste do hook de pre-commit, feito a pedido do usuário, e o commit que desfaria isso foi cancelado). Isso não bloqueia a escrita deste plano, mas **precisa ser corrigido antes de qualquer `/implementar`**, já que a Constitution exige typecheck e testes sempre verdes (Princípio 4).
