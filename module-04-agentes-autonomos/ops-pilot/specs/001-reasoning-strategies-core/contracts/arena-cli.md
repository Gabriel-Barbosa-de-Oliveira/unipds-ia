# Contract: Arena CLI (`src/arena.ts`, `npm run arena`)

## Invocação

```sh
npm run arena -- --input "<pergunta ou instrução operacional>" [--strategies react,plan-and-execute] [--max-iterations 8]
```

## Flags (validadas com zod antes de qualquer execução — Principle II da constitution)

| Flag | Obrigatória | Tipo/valores | Padrão |
|---|---|---|---|
| `--input` | Sim | string não vazia | — (erro de validação se ausente) |
| `--strategies` | Não | lista separada por vírgula, cada item ∈ `{ react, plan-and-execute }` | todas as estratégias registradas |
| `--max-iterations` | Não | inteiro positivo | 8 (mesmo padrão da estratégia Plan-and-Execute, FR-005) |

Flag inválida (ex.: `--strategies react,foo` ou `--max-iterations -1`) → a CLI encerra imediatamente com mensagem de erro clara e código de saída não-zero, sem executar nenhuma estratégia (FR-008 aplicado à fronteira de CLI).

## Comportamento

1. Valida as flags.
2. Para cada estratégia em `--strategies` (na ordem informada), invoca `strategy.run(input, { maxIterations })` (contrato em [reasoning-strategy.md](./reasoning-strategy.md)).
3. Imprime, para cada estratégia, um bloco identificado pelo nome contendo o trace passo a passo (formatado por `src/agents/trace.ts`) e as métricas (`llmCalls`, `latencyMs`).
4. Quando duas ou mais estratégias são executadas, imprime ao final um resumo comparativo (nome × `llmCalls` × `latencyMs`) — suporte direto à User Story 2 (comparar sem re-executar).

## Códigos de saída

| Cenário | Exit code |
|---|---|
| Todas as estratégias solicitadas executaram (mesmo que alguma tenha parado por limite de passos — isso é um resultado válido, não falha) | `0` |
| Flags inválidas | `1` |
| Falha de infraestrutura não recuperável (ex.: `OPENROUTER_API_KEY` ausente) antes de qualquer execução | `1` |
