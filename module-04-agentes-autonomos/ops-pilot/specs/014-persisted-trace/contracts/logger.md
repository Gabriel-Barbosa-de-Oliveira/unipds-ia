# Contract: logger JSON (`src/obs/logger.ts`)

As assinaturas abaixo são ilustrativas.

```ts
export type LogEvent = /* união fechada — ver data-model.md § LogEvent */;

/** Pura. Uma linha, sem quebras, com ts/level/event e os campos do evento. */
export function formatLogLine(event: LogEvent, now: Date): string;

/** Pura. Deriva os eventos de log de rota, fallback e tool de um trace, só com metadados. */
export function traceToLogEvents(requestId: string, trace: readonly TraceEvent[]): LogEvent[];

export interface Logger { log(event: LogEvent): void }

/** Padrão: escreve em process.stdout, uma linha por evento. */
export function createLogger(write?: (line: string) => void, now?: () => Date): Logger;
```

## Regras

- `formatLogLine` sempre produz um texto em que `JSON.parse` funciona e que não contém `\n` (SC-005).
- `traceToLogEvents`:
  - `route` vira `route.chosen`, sem `reason`;
  - `fallback` vira `model.fallback`, sem `reason`;
  - `action` vira `tool.called`, sem `args`;
  - todo o resto é ignorado.
- O `level` é fixo por tipo de evento (tabela do data-model).
- Nenhuma função do logger recebe um `Error` diretamente. Quem chama passa `errorType: error.name`, ou `"Error"` quando não houver nome.
- **Teste-âncora (SC-004)**: uma requisição completa (mensagem, resposta, args e resultados de tool, motivo do roteador e motivo de fallback contendo `MARCADOR-SECRETO-123`) não produz nenhuma linha de log contendo o marcador.
