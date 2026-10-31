# Tasks

> Sem artefato `specs` nesta change (decisão do usuário). As regras de validação que iriam para `specs/cfp-submission/spec.md` estão explicitadas abaixo, na task 3.1, e são a referência para os testes dos dois lados:
> - `name`: string, obrigatória, não vazia após trim, máx. 120 caracteres
> - `email`: string, obrigatória, formato de e-mail válido
> - `talkTitle`: string, obrigatória, não vazia após trim, máx. 200 caracteres
> - `isGDE`: booleano, obrigatório, sem coerção (`"true"` é rejeitado)
> - Qualquer outro campo, inclusive `id`, é rejeitado

## 1. Contrato compartilhado (`shared-types`)

- [x] 1.1 Exportar o tipo `CreateSpeakerSubmission = Omit<SpeakerDTO, 'id'>` em `shared-types/src/lib/speaker.dto.ts`, reexportado por `shared-types/src/index.ts`; verificar com `nx build shared-types` sem erros
- [x] 1.2 Remover o placeholder `sharedTypes()` e o seu spec, ou mantê-los com o teste passando; verificar com `nx test shared-types` e `nx lint shared-types` verdes

## 2. Setup do `api`

- [x] 2.1 Instalar `class-validator` e `class-transformer` (dependencies) e `supertest` + `@types/supertest` (devDependencies); verificar que `npm ls class-validator class-transformer supertest` resolve sem erros
- [x] 2.2 Extrair um `ValidationPipe` compartilhado (`whitelist: true`, `forbidNonWhitelisted: true`, `transform: true`, sem `enableImplicitConversion`) para um arquivo próprio em `api/src/app/` e aplicá-lo com `app.useGlobalPipes` no `api/src/main.ts`; verificar que `nx build api` compila e `nx serve api` sobe em `:3000`

## 3. Endpoint de submissão no `api`

- [x] 3.1 Criar `CreateSubmissionBody implements CreateSpeakerSubmission` (importado de `shared-types`), com os decorators do `class-validator` seguindo as regras do topo deste arquivo; verificar que remover um campo da classe quebra `nx build api` (prova do `implements`)
- [x] 3.2 Criar `SubmissionsService` com repositório em memória, que gera o `id` (ex.: `crypto.randomUUID()`) e devolve um `SpeakerDTO`; teste unitário Jest (`submissions.service.spec.ts`) verificando que o `id` é gerado, é único entre duas chamadas e que os demais campos são preservados
- [x] 3.3 Criar `SubmissionsController` com `@Post()` em `submissions` recebendo `@Body() body: CreateSubmissionBody` e retornando `201` com o `SpeakerDTO`; registrar o controller e o service num `SubmissionsModule` importado no `AppModule`; verificar com `curl -X POST localhost:3000/api/submissions` usando um payload válido → `201`
- [x] 3.4 Escrever `submissions.controller.spec.ts` (Jest + `Test.createTestingModule` + `useGlobalPipes` com o pipe compartilhado da 2.2 + `supertest`) cobrindo **400 Bad Request** para: corpo vazio; cada campo obrigatório ausente; `email` inválido; `name`/`talkTitle` vazios ou só com espaços; `name`/`talkTitle` acima do tamanho máximo; `isGDE` como `"true"` (string); campo extra (`id`, `foo`). Cobrir também **201** para o payload válido, com `id` presente na resposta. Verificar com `nx test api` verde
- [x] 3.5 Confirmar no grafo que `api -> shared-types` existe; verificar com `nx graph --file=stdout` (dependência `static` de `api` para `shared-types`)

## 4. Setup de testes do `frontend` (Vitest → Jest)

- [x] 4.1 Instalar `jest-preset-angular` numa versão compatível com Angular 22.1 e Jest 30; verificar com `npm ls jest-preset-angular` sem peer-deps não resolvidas
- [x] 4.2 Criar `frontend/jest.config.cts` (preset `jest-preset-angular`, `../jest.preset.js`), `frontend/src/test-setup.ts` e ajustar `frontend/tsconfig.spec.json` (`types: ["jest", "node"]`, incluir `test-setup.ts`); trocar o target `test` em `frontend/project.json` de `@angular/build:unit-test` para Jest (inferido pelo `@nx/jest/plugin` ou `@nx/jest:jest` explícito); verificar que `nx test frontend` executa o `app.spec.ts` existente com Jest e passa
- [x] 4.3 Adicionar `provideHttpClient()` em `frontend/src/app/app.config.ts`; verificar que `nx build frontend` compila

## 5. Formulário de submissão no `frontend`

- [x] 5.1 Criar `SubmissionService` (`HttpClient`, `POST /api/submissions`, entrada `CreateSpeakerSubmission`, saída `SpeakerDTO`, ambos de `shared-types`); teste Jest com `provideHttpClientTesting()` verificando método, URL e corpo da requisição
- [x] 5.2 Criar `SubmissionFormComponent` **standalone**, com estado em **Signals**: signals dos campos (`name`, `email`, `talkTitle` vazios; `isGDE` em `false`), `status` (`'idle' | 'submitting' | 'success' | 'error'`), `computed` de erros por campo (mesmas regras do topo) e `computed` `canSubmit = isValid && status !== 'submitting'`. Botão `<button type="submit" [disabled]="!canSubmit()">`; verificar que `nx build frontend` compila
- [x] 5.3 Aplicar **WAI-ARIA** no template: `<label for>` em cada campo; mensagem de erro com `id` ligada por `aria-describedby`; `aria-invalid="true"` em campo inválido e tocado; região `role="status"` + `aria-live="polite"` para "Enviando…", sucesso e erro do servidor (inclusive a mensagem do 400); `aria-busy` no `<form>` durante o envio; checkbox nativo para `isGDE`; verificar navegando só pelo teclado e checando a árvore de acessibilidade no DevTools
- [x] 5.4 Registrar a rota do formulário em `frontend/src/app/app.routes.ts` e remover o `nx-welcome` do `app.html`; verificar com `nx serve frontend` que a rota renderiza o formulário e que um envio válido chega ao `api` pelo proxy (`201` na aba Network)
- [x] 5.5 Escrever `submission-form.component.spec.ts` (Jest + `TestBed` + `provideHttpClientTesting()`) cobrindo: **estado inicial dos Signals** (campos vazios, `isGDE === false`, `status() === 'idle'`, `canSubmit() === false`); **botão de envio bloqueado** (`disabled`) no estado inicial e com dados inválidos (e-mail inválido, título vazio); botão habilitado com dados válidos; botão de novo `disabled` com `status === 'submitting'`; `aria-invalid` e `aria-describedby` presentes num campo inválido; resposta `400` levando `status` a `'error'` com a mensagem na região `aria-live`. Verificar com `nx test frontend` verde
- [x] 5.6 Confirmar no grafo que `frontend -> shared-types` existe; verificar com `nx graph --file=stdout`

## 6. Integração

- [ ] 6.1 Rodar `nx run-many -t lint test build -p shared-types api frontend` e verificar que tudo fica verde
- [ ] 6.2 Teste manual ponta a ponta com `nx serve frontend` (sobe o `api` junto): enviar um payload válido → sucesso anunciado; forçar um payload inválido pelo DevTools → `400` exibido na região `aria-live`
