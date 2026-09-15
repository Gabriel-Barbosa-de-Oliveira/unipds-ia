# Contract: `MemoryStore`/`SqliteMemoryStore` e `embeddings.ts` (`src/memory/`)

Ver [data-model.md](../data-model.md) para a forma das entidades e [research.md](../research.md) para as decisões por trás deste contrato.

## `src/memory/embeddings.ts`

```ts
function embed(text: string): Promise<Float32Array>;
```

- Singleton lazy: o pipeline `feature-extraction` (`onnx-community/all-MiniLM-L6-v2-ONNX`) só é carregado no primeiro `embed()` chamado — importar o módulo não baixa nem carrega o modelo.
- Chamada interna: `pipeline(text, { pooling: "mean", normalize: true })` — vetor de saída já normalizado (research.md item 1).
- Retorna sempre um `Float32Array` de 384 posições.

## Interface `MemoryStore` (`src/memory/memory-store.ts`)

```ts
interface RememberResult {
  stored: boolean; // false quando descartado por deduplicação (score > 0.92 contra um fato existente)
  id?: string;      // presente quando stored === true
}

interface RecallMatch {
  fact: string;
  score: number; // produto escalar, sempre >= 0.3
}

interface ForgetResult {
  removed: boolean;
  fact?: string; // o texto do fato removido, quando removed === true
}

interface MemoryStore {
  remember(userId: string, fact: string): Promise<RememberResult>;
  recall(userId: string, query: string, limit?: number): Promise<RecallMatch[]>;
  forget(userId: string, description: string): Promise<ForgetResult>;
}
```

## Construção (`SqliteMemoryStore`)

```ts
new SqliteMemoryStore(path?: string, embedFn?: (text: string) => Promise<Float32Array>)
```

- `path` default: `process.env.OPSPILOT_DB ?? "./data/opspilot.db"` — mesmo arquivo já usado por `SqliteOpsStore`/`SqliteConversationStore`, conexão própria e independente.
- `embedFn` default: `embed` de `src/memory/embeddings.ts` (o modelo real). Testes injetam uma função fake e determinística (research.md item 4) — só o teste dedicado de recuperação semântica genuína usa o `embed()` real.
- Conexão e DDL (`memories`, `CREATE TABLE IF NOT EXISTS`) são lazy — mesmo padrão de `SqliteOpsStore`/`SqliteConversationStore`.

## Métodos

Todos usam prepared statements; nenhuma query concatena valor de entrada na string SQL. `embedding` é lido/gravado como `BLOB` via `floatArrayToBuffer`/`bufferToFloatArray` (`src/domain/memory.ts`, research.md item 3).

### `remember(userId, fact): Promise<RememberResult>`

Calcula `embed(fact)`; compara (produto escalar, `src/domain/memory.ts#dotProduct`) contra o embedding de cada `Memory` existente do mesmo `userId`; maior score `> 0.92` → não grava, `{ stored: false }`; caso contrário → `INSERT`, `{ stored: true, id }`.

### `recall(userId, query, limit = 3): Promise<RecallMatch[]>`

Calcula `embed(query)`; compara contra cada `Memory` do `userId`; ordena por score decrescente, descarta score `< 0.3`, retorna os `limit` primeiros. `userId` sem fatos, ou nenhum acima do limiar → `[]` (nunca erro).

### `forget(userId, description): Promise<ForgetResult>`

Mesma busca de `recall`, sem limite de quantidade — toma o de maior score; `>= 0.3` → `DELETE` desse `Memory`, `{ removed: true, fact }`; abaixo disso (ou sem fatos) → `{ removed: false }`, nada é apagado.

## Tools do agente (`createMemoryTools`, `src/memory/memory-store.ts`)

```ts
function createMemoryTools(store: MemoryStore, userId: string): StructuredToolInterface[];
```

Duas tools, `remember_fact` e `forget_fact`, fechadas por closure sobre `store` **e** sobre `userId` (research.md item 6 — `userId` nunca é campo do schema zod, nunca preenchido pelo modelo). `recall` **não** é exposto como tool (research.md item 5) — é chamado diretamente pelo controller (`src/http/server.ts`).

| Tool | Schema (zod) | Comportamento |
|---|---|---|
| `remember_fact` | `{ fact: string }` | Chama `store.remember(userId, fact)`; retorna `{ stored, id? }` como observação |
| `forget_fact` | `{ description: string }` | Chama `store.forget(userId, description)`; retorna `{ removed, fact? }` como observação |

## Composição por requisição (`src/agents/index.ts`)

`resolveStrategy(name?, reflect?, extraTools?)` — terceiro parâmetro novo, opcional (research.md item 7). Quando o controller HTTP recebe `userId`, chama `resolveStrategy(strategy, reflect, createMemoryTools(memoryStore, userId))`; caso contrário, chama exatamente como antes (`resolveStrategy(strategy, reflect)`), sem nenhuma mudança de comportamento.
