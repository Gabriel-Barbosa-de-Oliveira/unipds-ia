# Contract: `src/memory/learning-reflector.ts`

Ver [data-model.md](../data-model.md) para a forma de `LearningVerdict` e [research.md](../research.md) para as decisões por trás deste contrato.

## `distillLearning(message: string): Promise<LearningVerdict>`

```ts
interface LearningVerdict {
  hasLearning: boolean;
  fact?: string;
}

function distillLearning(message: string): Promise<LearningVerdict>;
```

- Chama `createModel().withStructuredOutput(learningSchema).invoke(...)` (mesmo padrão de `src/agents/reflection.ts#critique`) com um prompt de sistema que instrui: identificar fatos duráveis sobre a pessoa ou seu contexto de trabalho; nunca tratar pedidos pontuais/perguntas como aprendizado; nunca incluir segredo/credencial em `fact`, mesmo que a mensagem contenha um fato genuíno ao lado (research.md item 6) — quando inseparável, `hasLearning: false`.
- Lança erro se o modelo não retornar um veredito estruturado válido (mesmo comportamento de `critique`, `reflection.ts`) — esse erro é responsabilidade de quem chama absorver (ver `reflectAndRemember`, abaixo).

## `reflectAndRemember(store, userId, message, distillFn?): Promise<void>`

```ts
type DistillFn = (message: string) => Promise<LearningVerdict>;

function reflectAndRemember(
  store: MemoryStore,
  userId: string,
  message: string,
  distillFn?: DistillFn, // default: distillLearning
): Promise<void>;
```

- **Nunca rejeita.** Qualquer erro de `distillFn` ou de `store.remember` é capturado internamente e vira apenas um `console.error` de diagnóstico (FR-006) — quem chama pode disparar sem `.catch` e sem `await`, com segurança.
- Chama `store.remember(userId, fact)` se, e somente se, `distillFn(message)` resolver com `hasLearning === true` e `fact` presente.
- `distillFn` é injetável (default: `distillLearning`, o modelo real) — testes injetam um fake determinístico, sem rede (research.md item 4, mesmo padrão de `critiqueFn` em `runReflectionLoop`).

## Ponto de disparo (`src/http/server.ts`)

`CreateAppOptions` ganha um campo novo, opcional:

```ts
interface CreateAppOptions {
  // ...campos existentes (007)
  /** Sobrescreve o refletor de aprendizado — usado por testes para injetar fakes, sem rede. */
  reflectAndRemember?: typeof reflectAndRemember;
}
```

Dentro do handler de `POST /chat`, só quando `userId` está presente na requisição e depois que `result` já foi calculado:

```ts
if (userId) {
  void reflect(memoryStore, userId, parsed.data.message).catch(() => {});
}
```

onde `reflect = options.reflectAndRemember ?? reflectAndRemember`. Chamada não aguardada — a resposta HTTP é montada e enviada em seguida, sem esperar por este processo (contracts/post-chat.md).
