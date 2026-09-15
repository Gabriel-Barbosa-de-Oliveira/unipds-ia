# Research: Memória Semântica

## 1. Modelo de embedding e biblioteca

**Decision**: `@huggingface/transformers` (transformers.js) v4, `pipeline("feature-extraction", "onnx-community/all-MiniLM-L6-v2-ONNX")`, chamado com `{ pooling: "mean", normalize: true }`; carregado como singleton lazy em `src/memory/embeddings.ts` (só instanciado no primeiro `embed()` chamado, nunca no import do módulo).

**Rationale**: transformers.js roda modelos ONNX localmente em Node, sem chamada de rede por inferência (só a chamada de rede é o download do checkpoint, uma vez, no primeiro uso — ver item 4). `onnx-community/all-MiniLM-L6-v2-ONNX` é o checkpoint ONNX do modelo `all-MiniLM-L6-v2` pedido, mantido pela própria organização `onnx-community` (Hugging Face) e é o exemplo oficial usado na documentação de tipos da versão instalada (`node_modules/@huggingface/transformers/types/pipelines/feature-extraction.d.ts`) — o checkpoint original (PyTorch, `sentence-transformers/all-MiniLM-L6-v2`) não roda diretamente nessa biblioteca. `pooling: "mean"` e `normalize: true` são exatamente o que o pedido original especifica.

**Alternatives considered**: chamar um serviço de embedding externo (OpenAI/OpenRouter embeddings). Rejeitada: contraria o pedido explícito ("local"), adiciona uma dependência de rede por chamada (custo e latência) e uma segunda credencial, quando a constitution já mandata SQLite local sem serviço externo para persistência — manter o embedding local também é consistente com esse espírito.

## 2. Métrica de similaridade

**Decision**: produto escalar bruto (`dotProduct`, `src/domain/memory.ts`) entre os vetores já normalizados (`normalize: true` no passo 1) — nunca uma fórmula de cosseno com normalização própria.

**Rationale**: o pedido original já especifica "produto escalar", e com `normalize: true` o produto escalar de dois vetores normalizados É numericamente idêntico à similaridade de cosseno — não há necessidade de dividir pelas normas de novo. Implementar a fórmula completa de cosseno seria trabalho redundante sobre vetores que já chegam normalizados.

## 3. Formato de armazenamento do vetor

**Decision**: `Float32Array` (saída de `embed()`) serializado para `Buffer` via `Buffer.from(vector.buffer)` e gravado na coluna `embedding BLOB` de `memories`; leitura faz o caminho inverso (`new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4)`). Funções puras de serialização (`floatArrayToBuffer`/`bufferToFloatArray`) em `src/domain/memory.ts`, ao lado de `dotProduct`.

**Rationale**: `all-MiniLM-L6-v2` produz vetores de 384 dimensões — `BLOB` binário é ~4x mais compacto que gravar como JSON/texto (384 floats como texto chegam a ~4-6KB por fato; como `Float32Array` binário, exatamente 1536 bytes) e evita parsing por linha lida. Mesmo padrão de "efeito colateral isolado, lógica pura ao lado" já usado pelo domínio das outras features.

## 4. Estratégia de teste (o modelo real é lento e precisa de rede na 1ª vez)

**Decision**: a lógica pura (produto escalar, seleção dos top-3 acima do limiar, serialização do vetor, deduplicação) é testada em `src/domain/memory.test.ts` com vetores forjados à mão — nunca invoca o modelo real, roda instantânea e offline. Um único teste dedicado, isolado, em `src/memory/embeddings.test.ts`, invoca o pipeline real para provar que duas frases sem nenhuma palavra em comum mas com o mesmo significado produzem um produto escalar acima do limiar de recall (0.3) — é o único ponto de todo o projeto que depende de rede (download do checkpoint ONNX na 1ª execução, cacheado localmente depois) e de tempo de inferência real (segundos, não milissegundos).

**Rationale**: um vetor forjado à mão não prova nada sobre entendimento semântico real — provar "recall acha fato sem palavra em comum" (o requisito central da feature) exige o modelo de verdade em pelo menos um teste; mas rodar o modelo real em toda a suíte de `SqliteMemoryStore` tornaria `npm test` lento e dependente de rede sempre, quebrando a suíte hoje 100% offline (114 testes, ~2.7s, `004`/`005`/`006`). Por isso `SqliteMemoryStore` recebe `embed` como dependência injetável (mesmo padrão de `resolveStrategy`/`conversationStore` das features 003/006): a implementação de produção usa o `embed()` real de `embeddings.ts`; a maior parte dos testes de `SqliteMemoryStore` (dedup, isolamento por `userId`, limite de 3, "não encontrado" no forget) injeta um `embed` fake e determinístico; só um teste de integração dedicado usa o `embed()` real para validar a promessa central da feature.

**Alternatives considered**: mockar a biblioteca inteira (`@huggingface/transformers`) em todos os testes. Rejeitada: isso tornaria impossível testar a promessa central da feature (recall sem palavra em comum) com qualquer confiança real — um mock não tem noção de significado.

## 5. `remember`/`forget` são tools do agente; `recall` é automático, não é tool

**Decision**: `remember_fact`/`forget_fact` são novas tools LangChain (`createMemoryTools(memoryStore, userId)`, em `src/memory/memory-store.ts`), que o modelo decide chamar durante o raciocínio — mesmo padrão de `open_incident`/`resolve_incident`. `recall`, ao contrário, roda automaticamente no controller HTTP (`src/http/server.ts`) antes de chamar a estratégia, sobre a mensagem recebida — nunca é uma tool que o modelo escolhe (ou não) chamar.

**Rationale**: "lembrar um fato" e "esquecer um fato" são ações que fazem sentido como decisão do modelo dentro do raciocínio (equivalente a "abrir um incidente"), então tools se encaixam bem. Já "recuperar contexto relevante antes de responder" precisa acontecer sempre, de forma confiável — se fosse uma tool, o cenário de teste da spec ("recall acha fato sem palavra em comum") dependeria do modelo decidir chamar a tool certa, o que não é garantido; automatizar no controller (mesmo padrão do histórico de conversa, `006-conversation-history`) torna o comportamento determinístico e sempre testável.

## 6. `userId` nunca é um parâmetro de tool — é capturado por closure a partir da requisição HTTP

**Decision**: `createMemoryTools(memoryStore, userId)` recebe `userId` como argumento da fábrica (fechado por closure nas duas tools), nunca como campo do schema zod que o modelo preenche.

**Rationale**: se `userId` fosse um parâmetro que o modelo decide preencher ao chamar `remember_fact`/`forget_fact`, um erro (ou uma alucinação) do modelo poderia gravar ou apagar um fato sob o `userId` errado — quebrando o isolamento entre pessoas (spec FR-007, SC-005), a garantia mais crítica da feature. Capturar `userId` da requisição HTTP (nunca do texto gerado pelo modelo) elimina essa classe de erro por construção.

## 7. Tools por requisição, sem tocar `react.ts`/`plan-and-execute.ts`/`reflection.ts`

**Decision**: `resolveStrategy` (`src/agents/index.ts`) ganha um terceiro parâmetro opcional, `extraTools?: StructuredToolInterface[]`; quando presente e não vazio, constrói a estratégia via `createReactStrategy([...opsTools, ...extraTools])`/`createPlanAndExecuteStrategy([...opsTools, ...extraTools])` (fábricas que **já existem** desde `004-ops-persistence`, usadas hoje por `bench.ts`) em vez de reaproveitar o singleton `reactStrategy`/`planAndExecuteStrategy`; quando omitido (todo chamador existente: `bench.ts`, `arena.ts`, e o próprio `/chat` quando não há `userId`), o comportamento é idêntico ao de hoje — mesmo singleton, zero mudança observável.

**Rationale**: `remember_fact`/`forget_fact` precisam do `userId` da requisição atual — o singleton `reactStrategy`/`planAndExecuteStrategy` é construído uma única vez, no carregamento do módulo, e não pode carregar um `userId` de uma requisição específica. Como `createReactStrategy(tools)`/`createPlanAndExecuteStrategy(tools)` já aceitam qualquer array de tools (é assim que `bench.ts` compõe seu próprio conjunto isolado), estender `resolveStrategy` para usá-las com um array extra é aditivo — nenhuma mudança em `react.ts`, `plan-and-execute.ts` ou `reflection.ts` (mesmo espírito de `006-conversation-history`, research.md item 1: minimizar o raio de mudança sobre as 3 estratégias já estáveis).

**Alternatives considered**: reconstruir tudo (`resolveStrategy` inteiro) por requisição, sempre. Rejeitada: reintroduziria custo de composição desnecessário no caminho sem `userId` (a maioria dos `/chat` de hoje), só para um caso que precisa de tools extras.

## 8. Identificação do fato a esquecer

**Decision**: `forget(userId, description)` reaproveita a mesma busca semântica do `recall` — calcula o produto escalar entre o embedding de `description` e cada fato daquele `userId`, toma o de maior score; se esse score for `>= 0.3` (mesmo limiar do recall), remove esse fato e informa qual foi removido; abaixo disso, não remove nada e informa que não encontrou um fato correspondente com confiança suficiente (spec FR-009, Assumptions).

**Rationale**: reaproveitar o mesmo limiar e a mesma função de busca evita introduzir um segundo número mágico não pedido pelo usuário original; e dá ao "esquecer" a mesma robustez semântica do "lembrar/recuperar" (esquecer também não deveria depender de palavras exatas).

## 9. Nova dependência e cache do checkpoint

**Decision**: adicionar `@huggingface/transformers` a `dependencies` (`package.json`); configurar `cache_dir: "./.cache/transformers"` explicitamente na chamada de `pipeline(...)` em `src/memory/embeddings.ts`; adicionar `.cache/` ao `.gitignore`.

**Rationale**: sem um `cache_dir` explícito, o local de cache padrão da biblioteca varia por ambiente/versão — fixar o caminho torna o comportamento previsível e fácil de excluir do controle de versão (mesmo espírito de `data/` já ignorado para os bancos SQLite). O primeiro `embed()` de cada ambiente novo baixa o checkpoint ONNX (~90MB) do Hugging Face Hub; execuções seguintes reusam o cache local, sem rede.
