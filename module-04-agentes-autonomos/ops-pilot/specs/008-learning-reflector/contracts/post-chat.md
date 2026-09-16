# Contract: `POST /chat` (`src/http/server.ts`, `createApp()`) — após `008-learning-reflector`

Estende o contrato de [`007-semantic-memory/contracts/post-chat.md`](../../007-semantic-memory/contracts/post-chat.md) (permanece a referência histórica daquela feature). **Nenhum campo de requisição ou resposta muda** — esta feature adiciona só um side-effect assíncrono, não observável na resposta HTTP.

## Requisição e resposta

Idênticas ao contrato de `007-semantic-memory` — mesmos campos (`message`, `strategy`, `reflect`, `conversationId`, `userId`), mesmos status (`200`/`400`/`404`/`422`/`504`/`500`), mesmo formato de corpo.

## Comportamento novo (side-effect, não observável na resposta)

Quando `userId` é informado, **depois** que o sistema já terminou de calcular a resposta da estratégia de raciocínio (mas sem esperar por este passo antes de responder à pessoa usuária):

1. A última mensagem da pessoa (`message` desta requisição) é analisada em busca de um fato durável elegível (`LearningVerdict`, ver [data-model.md](../data-model.md)).
2. Se um fato elegível for identificado, ele é registrado via `MemoryStore.remember(userId, fact)` — mesma chamada, mesma deduplicação, que a tool `remember_fact` (`007`) já usa quando o modelo decide chamá-la explicitamente.
3. Se nada for identificado (pedido pontual, pergunta, ou conteúdo sensível), ou se esse processo falhar por qualquer motivo, nada é registrado e nada disso é visível na resposta `200` — `answer`/`trace`/`metrics`/`conversationId` continuam exatamente no formato de `007`.

## Regras do contrato

- Sem `userId`: comportamento idêntico ao de antes desta feature — nenhuma análise, nenhuma chamada de modelo extra (mesma regra já valia para `recall`/`remember_fact`/`forget_fact` em `007`).
- A resposta HTTP nunca espera por este processo: o tempo até `200` (ou qualquer outro status) não é afetado por ele, mesmo que ele ainda esteja em andamento quando a resposta já foi enviada (FR-005, SC-003).
- Uma falha neste processo nunca vira um erro HTTP, nunca aparece no corpo da resposta, e nunca impede a resposta de ser enviada (FR-006).
- Um fato registrado por este processo é removível pela pessoa usuária exatamente pela mesma via já disponível para fatos ensinados manualmente — a tool `forget_fact` (`007`), sem nenhuma tool nova (research.md item 2).
