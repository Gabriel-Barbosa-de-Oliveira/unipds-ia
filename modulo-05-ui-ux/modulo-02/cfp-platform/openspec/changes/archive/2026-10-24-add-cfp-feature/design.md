# Design

## Context

Motivação e escopo: ver `proposal.md`. Estado atual do workspace que molda a abordagem:

- **Layout plano**: os projetos ficam em `api/`, `frontend/` e `shared-types/` na raiz. Não existe `apps/` nem `libs/`. O pedido original citava `apps/frontend` e `apps/api`; este design usa os caminhos reais.
- **Versões**: Angular **22.1** instalado (o pedido citava Angular 21). Standalone Components e Signals (`signal`, `computed`, `input`) existem nas duas versões, então nada no design depende da diferença.
- **`shared-types`**: o `SpeakerDTO` é uma `interface` (`id`, `name`, `email`, `talkTitle`, `isGDE`), resolvida pelo alias `shared-types` no `tsconfig.base.json`. Interfaces somem em runtime.
- **`api`**: NestJS 11 com `setGlobalPrefix('api')`, porta 3000, sem `ValidationPipe` e sem `class-validator` instalado. Testes com Jest (`ts-jest`) já configurados.
- **`frontend`**: o target `test` usa `@angular/build:unit-test` com Vitest (`vitest/globals` no `tsconfig.spec.json`). Não tem `provideHttpClient`. O `serve` depende de `api:serve` e faz proxy de `/api` para `:3000`.

## Goals / Non-Goals

**Goals:**
- O `SpeakerDTO` da `shared-types` é a única fonte do formato de dados nos dois apps; uma mudança no contrato quebra a compilação dos dois.
- A validação estrita roda no servidor e é a autoridade final; a validação no cliente serve só à UX.
- Os testes exigidos (400 no `api`; estado inicial do Signal e botão bloqueado no `frontend`) rodam com `nx test api` e `nx test frontend` usando Jest.

**Non-Goals:**
- Persistência real: as submissões ficam num repositório em memória, dentro do serviço.
- Usar decorators do `class-validator` dentro da `shared-types`. A lib continua sem dependências e utilizável pelo browser.
- Testes e2e (Playwright / `api-e2e`). Os projetos e2e existentes não mudam.
- i18n das mensagens de erro.

## Decisions

### D1. Contrato: `SpeakerDTO` como fonte da verdade, com tipo derivado para a entrada
- A requisição carrega `Omit<SpeakerDTO, 'id'>` (o cliente não escolhe o `id`). A resposta `201` devolve um `SpeakerDTO` completo, com `id` gerado pelo servidor.
- O tipo de entrada fica exportado da `shared-types` (ex.: `CreateSpeakerSubmission = Omit<SpeakerDTO, 'id'>`) para os dois lados usarem o mesmo nome.
- *Alternativa descartada*: o cliente enviar o `id`. Abre colisão de ids e não tem utilidade para o cliente.

### D2. Validação no `api`: classe no próprio app que `implements` o contrato
- O `class-validator` precisa de uma **classe** com decorators, e o `SpeakerDTO` é uma interface. Por isso o `api` define `CreateSubmissionBody implements CreateSpeakerSubmission`, com os decorators nela. Se a classe deixar de cumprir o contrato, o compilador acusa o drift.
- O controller recebe `@Body() body: CreateSubmissionBody`.
- `ValidationPipe` global em `main.ts` com `whitelist: true`, `forbidNonWhitelisted: true` e `transform: true`. Assim, campos extras (inclusive `id`) também viram `400`.
- O pipe é extraído para uma função/const compartilhada entre o `main.ts` e os testes, para que os testes exercitem a mesma configuração usada em produção.
- *Alternativas descartadas*: (a) transformar o `SpeakerDTO` em classe com decorators na `shared-types`, o que puxaria `class-validator` para o bundle do browser e acoplaria a lib ao Nest; (b) Zod, que vai contra a regra explícita de usar `class-validator`.

### D3. Regras de validação
As regras ficam nos specs (`cfp-submission`). Resumo do que o design assume: `name` e `talkTitle` como strings não vazias após trim e com tamanho máximo; `email` em formato válido; `isGDE` booleano obrigatório e sem coerção de `"true"`. A coerção fica desligada: com `transform: true`, as opções `enableImplicitConversion` continuam `false`.

### D4. Frontend: componente standalone com estado em Signals
- `SubmissionFormComponent` standalone, com uma rota própria em `app.routes.ts`.
- O estado fica em signals: um signal por campo (ou um signal de objeto `CreateSpeakerSubmission`), um `status` (`'idle' | 'submitting' | 'success' | 'error'`), um `computed` de erros por campo e um `computed` `canSubmit = isValid && status !== 'submitting'`.
- O botão usa `[disabled]="!canSubmit()"`. O estado inicial, que os testes verificam, é: campos vazios, `isGDE = false`, `status = 'idle'` e `canSubmit() === false`.
- O envio vai por um `SubmissionService` (`HttpClient`, `POST /api/submissions`), e o `provideHttpClient()` entra no `app.config.ts`.
- *Alternativa descartada*: Reactive Forms + `FormGroup`. Funciona, mas o pedido exige Signals para o estado, e misturar os dois cria duas fontes da verdade. Template-driven, com binding manual para signals, é suficiente para 4 campos.

### D5. Acessibilidade (WAI-ARIA)
- Cada campo tem um `<label for>` associado. Erros ficam em um elemento com `id` referenciado por `aria-describedby`, e o campo recebe `aria-invalid="true"` quando está inválido e já foi tocado.
- Uma região `role="status"` / `aria-live="polite"` anuncia "Enviando…", sucesso e erro do servidor. O formulário recebe `aria-busy` durante o envio.
- O `isGDE` é um `<input type="checkbox">` nativo, sem `role` customizado. O botão é um `<button type="submit">` nativo com `disabled`.
- Preferir HTML semântico nativo a atributos ARIA redundantes (primeira regra do ARIA).

### D6. Runner de testes do `frontend`: migrar para Jest
- O pedido exige Jest, e o workspace já tem o `@nx/jest/plugin`, o `jest.preset.js` e o Jest 30. O target `test` do `frontend` troca `@angular/build:unit-test`/Vitest por Jest com `jest-preset-angular`: `jest.config.cts`, `src/test-setup.ts` e `tsconfig.spec.json` com `types: ["jest"]`.
- O `app.spec.ts` existente precisa continuar passando depois da migração.
- *Alternativa descartada*: manter o Vitest. Ele é o padrão do gerador Angular deste workspace (`unitTestRunner: vitest-angular` no `nx.json`) e tem API quase idêntica, mas vai contra a regra explícita do pedido. Se a regra mudar, só o grupo de setup do `frontend` em `tasks.md` muda.

### D7. Estrutura de testes
- **`api`**: testes de controller com `Test.createTestingModule` + `app.useGlobalPipes(<pipe compartilhado>)` + `supertest`, cobrindo `400` para campo ausente, e-mail inválido, tipo errado, string vazia e campo extra, além de `201` para o payload válido. Um teste unitário do serviço cobre a geração do `id`.
- **`frontend`**: `TestBed` com o componente standalone e `provideHttpClientTesting()`. Os testes verificam os signals iniciais, o botão `disabled` no estado inicial e com dados inválidos, o botão habilitado com dados válidos, o botão de novo `disabled` durante `submitting`, e os atributos ARIA principais.

## Risks / Trade-offs

- [`jest-preset-angular` atrasado em relação ao Angular 22.1 ou ao Jest 30] → Fixar uma versão compatível na task de setup e validar com o `app.spec.ts` existente antes de escrever testes novos. Se não houver versão compatível, voltar à alternativa de D6 (Vitest) com aprovação do usuário.
- [Drift entre a classe validada e a interface] → O `implements` garante que os campos existem, mas não garante que as regras sejam iguais às do cliente. Mitigação: as regras ficam documentadas nos specs, e os testes dos dois lados usam os mesmos payloads de exemplo.
- [O repositório em memória perde os dados a cada restart] → Aceito para este escopo (Non-Goal). Fica atrás de um serviço para ser trocado depois.
- [Divergência entre validação do cliente e do servidor] → O servidor é a autoridade. O cliente exibe a mensagem do `400` na região `aria-live` quando o servidor rejeita algo que o cliente aceitou.

## Migration Plan

Não há dados nem consumidores existentes. O deploy é aditivo: endpoint e rota novos. Para rollback, basta reverter o commit. A troca do runner de testes do `frontend` é a única mudança que afeta código existente (`app.spec.ts`).
