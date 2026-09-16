# Data Model: Refletor de Aprendizado

Esta feature não introduz nenhuma tabela nova nem altera o schema de `memories` (`007-semantic-memory`, ver [007/data-model.md](../007-semantic-memory/data-model.md)). O único dado novo é um valor transiente, produzido e consumido dentro de um único ciclo de reflexão — nunca gravado por si só.

## LearningVerdict (transiente — nunca persistido)

| Campo | Tipo | Regras |
|---|---|---|
| `hasLearning` | boolean | `true` quando a última mensagem da pessoa contém um fato durável elegível; `false` para pedidos pontuais, perguntas, ou mensagens contendo segredo/credencial (mesmo que junto de um fato genuíno — research.md item 6) |
| `fact` | string, opcional | Presente somente quando `hasLearning === true`; texto do fato já destilado, no mesmo formato em linguagem natural aceito por `remember_fact` (`007`) |

Produzido por `distillLearning(message: string): Promise<LearningVerdict>` (`src/memory/learning-reflector.ts`), a partir da última mensagem da pessoa usuária (`parsed.data.message` em `POST /chat`) — nunca do histórico completo da conversa nem da resposta do copiloto.

## Regra de decisão: `reflectAndRemember(store, userId, message, distillFn?)`

1. Chama `distillFn(message)` (produção: `distillLearning`; testes: fake injetado — research.md item 4).
2. Se `hasLearning === true` **e** `fact` estiver presente → chama `store.remember(userId, fact)` (mesmo `MemoryStore.remember` de `007`, com sua própria deduplicação por score `> 0.92` — nenhuma lógica de dedup nova aqui).
3. Se `hasLearning === false`, ou `fact` ausente → nenhuma chamada a `store.remember`.
4. Qualquer erro em `distillFn` ou em `store.remember` é capturado e absorvido (FR-006, research.md item 3) — a função nunca rejeita.

Não há passo de leitura: esta feature nunca lê da tabela `memories` (isso continua sendo responsabilidade de `recall`/`forget`, inalterados desde `007`).

## Relacionamento com `Memory` (`007-semantic-memory`)

```text
LearningVerdict (transiente) ──[quando hasLearning]──> Memory.remember(userId, fact)
```

Um `Memory` criado a partir de um `LearningVerdict` é, a partir do momento em que é gravado, **idêntico** a um `Memory` criado via `remember_fact` (tool, `007`) — mesma tabela, mesmas colunas, mesma elegibilidade para `recall`/`forget`/`forget_fact`. Não existe campo de "origem" (automático vs. manual): a spec (Assumptions) exige explicitamente que a remoção funcione da mesma forma para os dois casos (FR-008), então distinguir a origem no dado persistido não teria uso.

## Contrato HTTP (`POST /chat`) — sem campos novos

Ver [contracts/post-chat.md](./contracts/post-chat.md). Nenhum campo de requisição ou resposta é adicionado, removido ou renomeado — o efeito desta feature é inteiramente um side-effect assíncrono, não observável na resposta HTTP em si.
