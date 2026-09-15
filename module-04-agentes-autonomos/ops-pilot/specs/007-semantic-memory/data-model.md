# Data Model: Memória Semântica

Entidade nova desta feature — bounded context distinto de dados operacionais (`004`) e de conversa (`006`). Campos e tipos são descritos de forma independente de implementação; o mapeamento para colunas SQL concretas é detalhe de tarefa, guiado pelas decisões de [research.md](./research.md).

## Memory (fato)

| Campo | Tipo | Regras |
|---|---|---|
| `id` | string | Identificador estável, gerado em `remember()` (`crypto.randomUUID()`) |
| `userId` | string | Obrigatório; dono do fato — nunca exposto/alterável por outra pessoa (research.md item 6) |
| `fact` | string | Texto literal do fato, em linguagem natural |
| `embedding` | vetor de 384 números (ponto flutuante) | Gerado a partir de `fact` (research.md item 1); usado para deduplicação e busca — nunca exposto pela API, só consumido internamente pelo store |
| `createdAt` | datetime | Timestamp de criação |

### Regra de escrita: `remember(userId, fact)`

1. Calcula o embedding de `fact`.
2. Calcula o produto escalar (research.md item 2) contra o embedding de cada fato já existente daquele `userId` (nunca de outro `userId` — isolamento, FR-007).
3. Se o maior desses scores for `> 0.92` → **não** grava (é considerado o mesmo fato, já registrado) — retorna indicação de que era uma duplicata.
4. Caso contrário → grava um `Memory` novo — retorna o `id` gerado.

### Regra de leitura: `recall(userId, query, limit = 3)`

1. Calcula o embedding de `query`.
2. Calcula o produto escalar contra o embedding de cada fato daquele `userId` (nunca de outro `userId`).
3. Ordena por score decrescente, descarta qualquer fato com score `< 0.3`, retorna os `limit` primeiros (no máximo 3, por padrão).
4. Nenhum fato relevante (todos abaixo de 0.3, ou `userId` sem fatos) → lista vazia, não é erro (spec Edge Case).

### Regra de escrita: `forget(userId, description)`

1. Mesma busca de `recall`, mas sem limite de quantidade — toma apenas o fato de maior score entre os daquele `userId`.
2. Score `>= 0.3` → remove esse `Memory`, retorna qual fato foi removido.
3. Score `< 0.3` (ou nenhum fato para esse `userId`) → não remove nada, retorna indicação de que nenhum fato correspondente foi encontrado (spec FR-009).

## Relacionamentos

```text
Pessoa de plantão (userId) 1 ── * Memory   (uma pessoa tem zero ou mais fatos registrados)
```

Não há relação entre `Memory` e `Conversation`/`ConversationMessage` (`006-conversation-history`) — são bounded contexts independentes; um fato registrado permanece disponível entre conversas diferentes da mesma pessoa (spec Assumptions).

## Repositório (`MemoryStore`) — superfície desta feature

| Método | Regras |
|---|---|
| `remember(userId: string, fact: string): Promise<{ stored: boolean; id?: string }>` | Ver regra de escrita acima. `stored: false` quando descartado por deduplicação. |
| `recall(userId: string, query: string, limit?: number): Promise<{ fact: string; score: number }[]>` | Ver regra de leitura acima. Lista vazia é resultado válido. |
| `forget(userId: string, description: string): Promise<{ removed: boolean; fact?: string }>` | Ver regra de escrita acima. `removed: false` quando nada corresponde com confiança suficiente. |

## Contrato HTTP (`POST /chat`) — campos novos

Ver [contracts/post-chat.md](./contracts/post-chat.md) para o contrato completo do endpoint após esta feature.

| Campo | Onde | Regra |
|---|---|---|
| `userId` (requisição) | body, opcional | Quando informado, ativa a memória semântica para esta requisição: `recall` automático antes de responder, e as tools `remember_fact`/`forget_fact` ficam disponíveis ao modelo, ambas escopadas a esse `userId`; quando omitido, o comportamento é idêntico ao de antes desta feature (nenhuma memória envolvida) |
