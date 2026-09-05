# Research: Endpoint HTTP de Chat

Nenhum item do Technical Context ficou como `NEEDS CLARIFICATION` — as decisões abaixo documentam escolhas de design que tinham mais de uma alternativa razoável.

## 1. Framework HTTP

**Decision**: Express (`express` já é dependência de produção do projeto e é mandatado pela constitution como servidor HTTP).

**Rationale**: Nenhuma decisão real a tomar — a stack tecnológica obrigatória já fixa Express; introduzir outro framework HTTP exigiria amendment da constitution, fora de escopo desta feature.

**Alternatives considered**: N/A.

## 2. Timeout: encerrar a resposta ao cliente vs. cancelar a execução em andamento

**Decision**: `runWithTimeout` corre `strategy.run(...)` contra um timer via `Promise`; se o timer vence, a promise retornada rejeita com `ChatTimeoutError` e o controller responde 504 imediatamente. A chamada `strategy.run(...)` que perdeu a corrida **não é cancelada à força** — ela é apenas descartada do ponto de vista do cliente (seu resultado, se chegar depois, não é usado nem observável).

**Rationale**: FR-008 e a User Story 4 garantem que o **cliente** nunca fica esperando além do limite — é isso que os acceptance scenarios verificam (cliente recebe erro de timeout dentro de uma margem razoável). Cancelamento de verdade da chamada ao modelo em andamento exigiria propagar um `AbortSignal` por `RunOptions` e implementá-lo dentro de `react.ts`, `plan-and-execute.ts` e `reflection.ts` (já entregues pelas features 001/002) — uma mudança bem maior que o escopo desta feature (só expor via HTTP o que já existe) e desproporcional ao problema que o timeout resolve aqui.

**Alternatives considered**: (a) estender `RunOptions` com `signal?: AbortSignal` e implementar cancelamento cooperativo dentro de cada estratégia — rejeitada por escopo/risco (tocaria arquivos já validados das features 001/002 sem necessidade); (b) matar o processo/worker — rejeitada, blast radius absurdo para uma única requisição lenta.

## 3. Formato do registro de estratégias e composição com reflection

**Decision**: `src/agents/index.ts` mantém um registro apenas com as estratégias **base** (`react`, `plan-and-execute`); o parâmetro `reflect` da requisição é um flag ortogonal aplicado em tempo de resolução via `withReflection(strategy)` já existente — não uma entrada adicional do registro.

**Rationale**: É o contrato literal pedido para a feature (registry nome→estratégia; "reflect aplica withReflection"). Também evita crescimento combinatório do registro: cada estratégia nova automaticamente ganha suporte a `reflect: true` sem precisar de uma segunda entrada nomeada.

**Alternatives considered**: reaproveitar o registro "achatado" de `src/arena.ts` (`"react"`, `"plan-and-execute"`, `"reflect:react"`, `"reflect:plan-and-execute"` como 4 nomes independentes) — rejeitada para o endpoint HTTP porque o contrato pedido pelo usuário já separa `strategy` de `reflect` como dois campos independentes do corpo da requisição (FR-001, FR-006).

## 4. Injeção de dependência para testabilidade determinística

**Decision**: `createApp(options?)` aceita `resolveStrategy` (a função nome→estratégia completa, não apenas o registro) e `timeoutMs` como overrides opcionais. O teste de integração do endpoint passa um `resolveStrategy` fake que nunca invoca `withReflection`/`createModel` reais, e um `timeoutMs` pequeno para exercitar o cenário 504 sem esperar 180s de verdade.

**Rationale**: É o requisito explícito de teste da feature ("teste de integração com estratégia fake determinista, sem rede"). Injetar a função de resolução inteira (não só o mapa de estratégias) garante que mesmo o caminho `reflect: true` seja testável sem rede — se apenas o registro base fosse injetável, `reflect: true` ainda chamaria o `withReflection` real, que internamente chama `createModel()` ao executar.

**Alternatives considered**: mockar `createModel()`/módulo do LangChain diretamente — rejeitada; o projeto não usa biblioteca de mocking e a convenção já estabelecida (`reflection.ts`/`reflection.test.ts`) é injeção de fakes puros, não mock de módulo.

## 5. Ferramenta para o teste de integração HTTP

**Decision**: subir o `Express` app real em porta efêmera (`app.listen(0)`) e usar `fetch` nativo do Node para fazer as requisições no teste; fechar o servidor (`server.close()`) ao final.

**Rationale**: Node 24 já inclui `fetch` e `http.Server` nativamente — nenhuma dependência de teste nova é necessária para um único endpoint. Mantém o princípio de não introduzir dependências além do necessário.

**Alternatives considered**: adicionar `supertest` como devDependency — rejeitada por desnecessária para o escopo atual (um endpoint); pode ser revisitada se o número de endpoints crescer.

## 6. Não tocar `src/arena.ts`

**Decision**: `src/arena.ts` continua com sua própria definição local de `STRATEGIES` (incluindo as entradas `reflect:react`/`reflect:plan-and-execute`), sem importar o novo `src/agents/index.ts`.

**Rationale**: A duplicação é pequena (2 linhas mapeando `react`/`plan-and-execute` para suas estratégias) e o ganho de reuso não compensa o risco de alterar um arquivo já entregue e validado pelas features 001/002 como parte de uma feature cujo escopo é estritamente aditivo (expor um novo endpoint HTTP). Mantém o raio de impacto desta feature contido em arquivos novos + `domain/errors.ts` + `src/index.ts` (Principle VIII).

**Alternatives considered**: refatorar `arena.ts` para importar `STRATEGIES`/`resolveStrategy` de `agents/index.ts` — deixado como possível follow-up, não incluído aqui.

## 7. Taxonomia de erro → status HTTP

**Decision**:

| Situação | Camada que detecta | Tipo | Status HTTP |
|---|---|---|---|
| Corpo da requisição malformado ou campos inválidos | Controller (`ChatRequestSchema.safeParse`) | Falha de validação de fronteira (zod `issues`), não uma classe de erro de domínio | 400 |
| `strategy` informado não corresponde a nenhuma estratégia registrada | `resolveStrategy` (`agents/index.ts`) | `UnknownStrategyError` | 422 |
| Execução ultrapassa `timeoutMs` | `runWithTimeout` (`services/chat.service.ts`) | `ChatTimeoutError` | 504 |
| Qualquer outra falha (ex.: `OPENROUTER_API_KEY` ausente, erro de rede do provedor do modelo) | Propagada de dentro da estratégia | Erro genérico não classificado | 500, mensagem genérica ao cliente, detalhe completo apenas no log de servidor |

**Rationale**: corpo inválido é uma falha de **formato** da requisição (bug do cliente antes mesmo de haver um "pedido" de domínio válido) — por isso é tratada diretamente no controller com os `issues` do zod, sem virar uma classe de erro de domínio, mesmo padrão implícito já usado no projeto (zod na fronteira, Principle II). `UnknownStrategyError`/`ChatTimeoutError` são falhas previsíveis de **negócio** (um recurso referenciado não existe; um limite de tempo foi atingido) e por isso seguem o Principle III (classes de erro de domínio, traduzidas só na borda). Falhas não previstas nunca vazam detalhe interno ao cliente (Principle VI).
