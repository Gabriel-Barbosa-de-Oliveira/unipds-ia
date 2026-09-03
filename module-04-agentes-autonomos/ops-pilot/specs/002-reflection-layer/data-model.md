# Data Model: Camada de Reflection

Entidades derivadas do [spec.md](./spec.md) (seção Key Entities) e das decisões de [research.md](./research.md). Nenhuma destas entidades é persistida — são todas construções efêmeras de uma única execução, na mesma linha do que a feature 001 já define para `ReasoningStrategy`/`TraceEvent`/`Metrics` em [../001-reasoning-strategies-core/data-model.md](../001-reasoning-strategies-core/data-model.md). Campos e tipos são descritos de forma independente de implementação (a codificação TypeScript/zod concreta é detalhe de tarefa).

## ReflectionOptions (contrato de construção, não persistido)

Parâmetros usados ao decorar uma estratégia com `withReflection`.

| Campo | Tipo | Regras |
|---|---|---|
| `maxReflections` | number (inteiro ≥ 0) | Opcional; padrão **2** quando omitido (FR-005). Conta regenerações extras após a 1ª tentativa — o número máximo de tentativas avaliadas em uma execução é `maxReflections + 1`. |

## Critique (contrato de execução, não persistido)

Resultado de uma avaliação crítica sobre uma tentativa de resposta.

| Campo | Tipo | Regras |
|---|---|---|
| `approved` | boolean | `true` encerra o ciclo com a tentativa avaliada; `false` dispara uma regeneração, se o limite ainda não foi atingido |
| `feedback` | string, não vazia | Justificativa da decisão; sempre presente, mesmo quando `approved` é `true` (FR-002) — é o conteúdo registrado no evento `critique` do trace e, quando reprovado, também o texto incorporado à próxima tentativa (`buildRetryInput`, ver research.md item 3) |

## Attempt (conceito interno, não persistido)

Uma execução completa da estratégia envolvida dentro do ciclo de reflection. Não é um tipo novo exposto publicamente — é o `RunResult` já definido pela feature 001 (`answer`, `trace`, `metrics`), produzido por uma chamada a `strategy.run(input, options)`.

| Campo | Tipo | Regras |
|---|---|---|
| `answer` | string | Resposta final desta tentativa; entra como "tentativa anterior" no `buildRetryInput` se for reprovada |
| `trace` | TraceEvent[] | Trace desta tentativa isolada, antes de ser concatenado ao histórico completo do ciclo (ver "Histórico de Reflection" abaixo) |
| `metrics` | Metrics | `llmCalls` desta tentativa isolada; somado ao total do ciclo junto com as chamadas do crítico (FR-008) |

## Histórico de Reflection (resultado agregado, é o RunResult final devolvido por `withReflection(...).run()`)

O `RunResult` de uma execução com reflection é o mesmo tipo já definido pela feature 001 — nenhuma extensão de tipo é necessária —, mas seu conteúdo é agregado a partir de múltiplas tentativas e críticas:

| Campo | Tipo | Regras |
|---|---|---|
| `answer` | string | A `answer` da última tentativa executada — aprovada pelo crítico, ou a melhor disponível quando `maxReflections` se esgota sem aprovação (FR-006) |
| `trace` | TraceEvent[] | Concatenação cronológica de: trace da tentativa 1, evento `critique` da 1ª avaliação, (se reprovada) trace da tentativa 2, evento `critique` da 2ª avaliação, e assim por diante — reindexado (`at`) para ordem crescente contínua (research.md item 5) |
| `metrics.llmCalls` | number | Soma de `metrics.llmCalls` de todas as tentativas mais 1 por avaliação crítica realizada (FR-008) |
| `metrics.latencyMs` | number | Medido de ponta a ponta pelo wrapper (início da 1ª tentativa até a decisão final), não a soma das latências individuais (research.md item 5) |

### Transições do ciclo

```text
tentativa 1 → crítica 1 ──approved──→ FIM (answer = tentativa 1)
                        └─reprovada, regenerações < maxReflections─→ tentativa 2 → crítica 2 → ...
                        └─reprovada, regenerações == maxReflections─→ FIM (answer = última tentativa, não aprovada)
```

- Número máximo de tentativas avaliadas: `maxReflections + 1` (padrão 3, quando `maxReflections = 2`).
- Número máximo de eventos `critique` no trace: igual ao número de tentativas avaliadas (uma crítica por tentativa, inclusive a última quando o limite é atingido sem aprovação).

## Relacionamentos

```text
ReasoningStrategy (feature 001) ──decorada por──> withReflection(strategy, opts: ReflectionOptions) ──produz──> ReasoningStrategy
                                                                                                                  (name = "reflect:" + strategy.name)

withReflection(...).run(input, options)
  → executa N Attempts (N = 1..maxReflections+1), cada um produzindo 1 RunResult (feature 001)
  → cada Attempt é seguido por exatamente 1 Critique
  → agrega tudo em 1 RunResult final (Histórico de Reflection acima)
```
