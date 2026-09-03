# Contract: Arena CLI — extensão para reflection (`src/arena.ts`, `npm run arena`)

Extensão aditiva do contrato já definido em [../../001-reasoning-strategies-core/contracts/arena-cli.md](../../001-reasoning-strategies-core/contracts/arena-cli.md). Nenhuma flag, comportamento ou código de saída existente muda — apenas o conjunto de nomes de estratégia aceitos cresce.

## Invocação

```sh
npm run arena -- --input "<pergunta ou instrução operacional>" \
  [--strategies react,plan-and-execute,reflect:react,reflect:plan-and-execute] \
  [--max-iterations 8]
```

## Flags (validadas com zod antes de qualquer execução — Principle II da constitution)

| Flag | Obrigatória | Tipo/valores | Padrão |
|---|---|---|---|
| `--input` | Sim | string não vazia | — (erro de validação se ausente) |
| `--strategies` | Não | lista separada por vírgula, cada item ∈ `{ react, plan-and-execute, reflect:react, reflect:plan-and-execute }` | todas as estratégias registradas (as quatro) |
| `--max-iterations` | Não | inteiro positivo | 8 — aplicado a **cada tentativa** de uma estratégia `reflect:*`, do mesmo jeito que já se aplica a uma execução única de `react`/`plan-and-execute` (FR-012) |

Flag inválida (ex.: `--strategies react,foo`) → a CLI encerra imediatamente com mensagem de erro clara e código de saída não-zero, sem executar nenhuma estratégia — comportamento inalterado do contrato base.

## Comportamento

1. Valida as flags (inalterado).
2. Para cada estratégia em `--strategies` (na ordem informada), invoca `strategy.run(input, { maxIterations })`. Para `reflect:react`/`reflect:plan-and-execute`, essa chamada dispara o ciclo completo de reflection (1 ou mais tentativas + críticas) descrito em [reflection-strategy.md](./reflection-strategy.md), mas o ponto de invocação a partir da arena é idêntico ao de qualquer outra estratégia — a arena não precisa saber que há reflection por trás do nome.
3. Imprime, para cada estratégia, um bloco identificado pelo nome (`reflect:react`, `reflect:plan-and-execute` incluídos) contendo o trace completo — incluindo os eventos `critique` intercalados entre tentativas — formatado por `src/agents/trace.ts`, e as métricas (`llmCalls`, `latencyMs`) já agregadas (tentativas + crítico).
4. Quando duas ou mais estratégias são executadas, imprime ao final um resumo comparativo (nome × `llmCalls` × `latencyMs`) — inalterado; permite comparar diretamente, por exemplo, `react` contra `reflect:react` na mesma linha (User Story 2).

## Códigos de saída

Inalterados em relação ao contrato base — nenhuma estratégia `reflect:*` introduz um novo cenário de saída; reprovação do crítico e esgotamento de `maxReflections` são resultados válidos (exit code `0`), não falhas.
