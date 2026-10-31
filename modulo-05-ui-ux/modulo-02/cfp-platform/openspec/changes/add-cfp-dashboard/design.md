# Design

## Context

- `api`: `SubmissionsController` (`@Controller('submissions')`, prefixo global `api`) só tem `@Post()`. `SubmissionsService` guarda as propostas num array privado em memória (`submissions: SpeakerDTO[]`), sem método de leitura. Testes de controller usam `Test.createTestingModule` + `supertest`.
- `frontend` (Angular 22.1 — o pedido diz 21; o workspace está em `~22.1.0` e nada aqui depende da diferença): `provideHttpClient()` e `provideRouter(appRoutes)` já estão em `app.config.ts`. Única rota: `''` → `SubmissionFormComponent` via `loadComponent`. O formulário é Standalone, OnPush, estado em `signal()`, e chama a API por `SubmissionService` (`providedIn: 'root'`). Testes com Jest + `HttpTestingController`.
- **Não existem Design Tokens hoje.** A identidade visual está em valores fixos no `submission-form.component.scss`: `#1d4ed8` (primário), `#9ca3af` (desabilitado), `#6b7280` (borda), `#b91c1c` (erro), `#15803d` (sucesso), `#2563eb` (foco, contorno 3px / offset 2px), raio `0.375rem`, fonte `system-ui, sans-serif`, host `max-width: 32rem; margin: 2rem auto; padding: 0 1rem`. `styles.scss` está vazio.
- Budget de estilo por componente: aviso em 4kb, erro em 8kb.

## Goals / Non-Goals

**Goals:**
- Uma única fonte de tokens consumida pelos dois componentes, com o formulário visualmente idêntico.
- Dashboard seguindo o mesmo padrão do formulário (Standalone, OnPush, Signals, testes Jest no mesmo estilo).

**Non-Goals:**
- Atualização automática/polling da lista; a lista é carregada ao abrir a página.
- Refatorar o formulário além de trocar valores fixos por tokens e incluir o link.
- Tema escuro ou novos tokens que o formulário não usa.

## Decisions

### 1. Estender o controller e o service existentes
`@Get()` em `SubmissionsController` chamando um novo `SubmissionsService.findAll(): SpeakerDTO[]`, que devolve uma cópia (`[...this.submissions]`) para que nenhum chamador altere o armazenamento por referência. A resposta mantém a ordem de inserção.
- *Alternativa*: criar um `CfpController` novo, como o pedido nomeia. Rejeitada: duplicaria a rota `submissions` e separaria leitura e escrita do mesmo recurso.

### 2. `CfpDashboardComponent` injeta `HttpClient` diretamente
Conforme o pedido: `inject(HttpClient)` no componente e `GET /api/submissions` disparado no construtor, com `takeUntilDestroyed()` na assinatura. Arquivos em `frontend/src/app/submissions/cfp-dashboard.component.{ts,html,scss,spec.ts}`, seletor `app-cfp-dashboard`, ao lado do formulário (mesma feature).
- *Alternativa*: adicionar `list()` ao `SubmissionService`, como o formulário faz com `submit()`. Seria mais consistente e centralizaria a URL; fica descartada só porque o pedido pede o `HttpClient` no componente. Trocar depois não muda specs nem testes de comportamento.
- *Alternativa*: `httpResource()`. Rejeitada: o pedido pede `WritableSignal<SpeakerDTO[]>`, e `httpResource` expõe um sinal só de leitura.

### 3. Estado em Signals
- `readonly speakers: WritableSignal<SpeakerDTO[]> = signal([])`
- `readonly status = signal<'loading' | 'loaded' | 'error'>('loading')`
- `computed`: `isEmpty` (`loaded` e lista vazia) e `statusMessage` (texto da região viva: "Carregando propostas…" / "N propostas enviadas." / mensagem de erro).
No `next`: `speakers.set(lista)` e `status.set('loaded')`; no `error`: `status.set('error')`, independentemente do status HTTP.

### 4. Tabela semântica, não cards
Os quatro campos são dados tabulares homogêneos e comparáveis por coluna; `<table>` dá a leitores de tela a navegação por linha/coluna de graça. Estrutura: `<section aria-labelledby>` → `h1` → região viva → `<div class="table-wrapper">` (`overflow-x: auto` para telas estreitas) → `<table>` com `<caption>`, `<thead>` de `th scope="col"` e `@for (speaker of speakers(); track speaker.id)`. Booleano exibido como "Sim"/"Não". `aria-busy` na `section` enquanto `status() === 'loading'`.
- *Alternativa*: lista de cards (`<ul>` de `<article>`). Rejeitada para este volume de campos; melhor só se a tela ganhar mais conteúdo por proposta.

### 5. Design Tokens como CSS custom properties globais
Declarar em `frontend/src/styles.scss`, em `:root`, com os valores atuais exatos:

| Token | Valor |
|---|---|
| `--cfp-font-family` | `system-ui, sans-serif` |
| `--cfp-color-primary` | `#1d4ed8` |
| `--cfp-color-on-primary` | `#fff` |
| `--cfp-color-disabled` | `#9ca3af` |
| `--cfp-color-border` | `#6b7280` |
| `--cfp-color-danger` | `#b91c1c` |
| `--cfp-color-success` | `#15803d` |
| `--cfp-color-focus` | `#2563eb` |
| `--cfp-radius` | `0.375rem` |
| `--cfp-focus-ring` | `3px solid var(--cfp-color-focus)` |

O SCSS do formulário passa a usar `var(--…)`; o do dashboard também. Custom properties atravessam o encapsulamento emulado do Angular, então não exigem `ViewEncapsulation.None` nem import em cada componente.
- *Alternativa*: variáveis/mixins SCSS num partial importado pelos dois componentes. Rejeitada: duplicaria o CSS compilado em cada componente (pesa no budget de 4kb) e não fica inspecionável no navegador.

### 6. Navegação com `routerLink` estilizado como botão primário
"Botão de navegação" vira `<a routerLink="/dashboard" class="button-link">` no formulário (fora do `<form>`, abaixo da região viva) e `<a routerLink="/" class="button-link">` no dashboard. Navegação é semanticamente um link; usar `<button>` quebraria abrir em nova aba e a expectativa de leitores de tela. A classe `.button-link` (mesmo padding, peso, raio, cor primária e foco do `button`) fica em `styles.scss` para não duplicar entre componentes. Os dois componentes importam `RouterLink`.
- Consequência: o spec do formulário passa a precisar de `provideRouter([])` no `TestBed`.

### 7. Rota
`{ path: 'dashboard', title: 'Propostas enviadas', loadComponent: () => import('./submissions/cfp-dashboard.component').then(m => m.CfpDashboardComponent) }`, no mesmo padrão lazy da rota existente.

### 8. Layout
O host do dashboard reaproveita margem, padding e fonte do formulário, mas com `max-width: 48rem` — 32rem não comporta quatro colunas com e-mail e título. Tabela com `border-collapse: collapse`, bordas de linha em `--cfp-color-border`, cabeçalho em peso 600 (como os `label` do formulário).

## Risks / Trade-offs

- [`GET /api/submissions` expõe nome e e-mail de todos os palestrantes sem autenticação] → Aceito porque autenticação está fora do escopo e o app não está em produção (`main.ts` já avisa isso). Registrar no proposal; qualquer deploy público precisa de uma change de autenticação antes.
- [Dados somem ao reiniciar o `api` (armazenamento em memória)] → Já é o comportamento atual; o dashboard só torna isso visível.
- [Extrair tokens pode alterar o visual do formulário sem querer] → Valores copiados literalmente; os testes do formulário continuam verdes e a task de verificação compara visualmente o formulário antes/depois.
- [URL `/api/submissions` duplicada entre `SubmissionService` e o dashboard] → Consequência da decisão 2; aceitável com dois usos.
- [Lista sem paginação] → Adequado ao volume de um CFP com armazenamento em memória.
