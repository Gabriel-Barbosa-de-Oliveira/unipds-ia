# Contract: `withReflection` decorator

Definido em `src/agents/reflection.ts`. Assinatura ilustrativa (a codificação final de tipos é detalhe de tarefa):

```ts
import type { ReasoningStrategy, RunOptions, RunResult } from "./types.ts";

interface ReflectionOptions {
  maxReflections?: number; // default 2
}

interface Critique {
  approved: boolean;
  feedback: string;
}

function withReflection(strategy: ReasoningStrategy, opts?: ReflectionOptions): ReasoningStrategy;

// Orquestração pura, testável com fakes — não chama createModel() nem faz IO diretamente.
type RunAttempt = (input: string, options?: RunOptions) => Promise<RunResult>;
type CritiqueFn = (attempt: RunResult, input: string) => Promise<Critique>;

function runReflectionLoop(
  runAttempt: RunAttempt,
  critique: CritiqueFn,
  input: string,
  options: RunOptions | undefined,
  maxReflections: number,
): Promise<RunResult>;

// Helpers puros de formatação, usados pelo wiring real dentro de withReflection.
function buildRetryInput(originalInput: string, previousAnswer: string, feedback: string): string;
function buildCritiquePrompt(input: string, trace: RunResult["trace"], answer: string): string;
```

## Regras do contrato

- `withReflection(strategy, opts)` DEVE retornar um `ReasoningStrategy` cujo `name` é `"reflect:" + strategy.name` (ex.: `reflect:react`, `reflect:plan-and-execute`) — nunca reescreve ou muta `strategy` em si; a estratégia original continua utilizável e inalterada após a decoração.
- `opts.maxReflections`, quando omitido, DEVE assumir o valor **2**. `maxReflections` conta regenerações extras após a 1ª tentativa: o número máximo de tentativas avaliadas por uma execução é `maxReflections + 1` (FR-005, esclarecido com o usuário na fase de spec).
- `run(input, options)` do decorador NUNCA lança para reprovação do crítico ou esgotamento de `maxReflections` — essas são condições de negócio esperadas que produzem um `RunResult` normal (FR-011), no mesmo espírito do contrato de `ReasoningStrategy` da feature 001. `run` só rejeita a Promise para falhas de infraestrutura não recuperáveis propagadas pela estratégia base ou pela chamada ao crítico (ex.: `OPENROUTER_API_KEY` ausente, erro de rede do modelo).
- `options` (incluindo `maxIterations`) é repassado inalterado a `strategy.run(...)` em **cada** tentativa — o limite de passos por tentativa da estratégia base nunca é relaxado nem reforçado pelo decorador (FR-012).
- Toda avaliação crítica realizada DEVE gerar exatamente um evento `{ type: "critique", at, content: feedback }` no trace final, na posição cronológica correspondente (após o trace da tentativa que ela avaliou, antes do trace de uma eventual próxima tentativa) — FR-007.
- `metrics.llmCalls` do `RunResult` final DEVE ser a soma de `metrics.llmCalls` de todas as tentativas mais exatamente 1 por avaliação crítica realizada — nunca reportar apenas a última tentativa nem omitir as chamadas do crítico (FR-008).
- `runReflectionLoop` é assíncrona mas determinística dado o mesmo `runAttempt`/`critique`/`input`/`options`/`maxReflections` — não instancia modelo, não lê `process.env`, não faz nenhuma chamada de rede diretamente; toda IO real fica isolada em `withReflection`, que conecta `strategy.run` como `runAttempt` e um crítico baseado em `createModel().withStructuredOutput(CritiqueSchema)` como `critique`.
