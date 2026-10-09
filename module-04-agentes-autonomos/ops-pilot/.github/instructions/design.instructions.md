---
applyTo: "web/**"
paths:
  - "web/**"
---

# Design — frontend (`web/`)

## Hierarquia

- Um único `h1` por página; títulos seguem a ordem `h1 → h2 → h3` sem pular níveis.
- Uma ação primária por tela/seção; ações secundárias com menor peso visual (outline/ghost).
- Hierarquia por tamanho, peso e cor, nessa ordem — não use só cor para diferenciar.
- Escala tipográfica fixa (ex.: 12 / 14 / 16 / 20 / 24 / 32 px); corpo em 16px, line-height entre 1.4 e 1.6.
- Informação crítica de plantão (severidade, status do incidente) sempre visível sem rolagem.

## Espaçamento em escala

- Use somente tokens de uma escala base 4px: `4, 8, 12, 16, 24, 32, 48, 64`.
- Proibido valor mágico (`13px`, `margin: 7px`); se faltar um valor, discuta antes de ampliar a escala.
- Espaço interno de um grupo < espaço entre grupos (proximidade indica relação).
- Prefira `gap` em flex/grid a margens em filhos.
- Tokens centralizados (CSS custom properties ou tema) — nunca hardcoded no componente.

## Estados vazios e de erro

Todo componente que carrega dados trata os quatro estados: **loading**, **vazio**, **erro** e **sucesso**.

- **Loading:** skeleton com o formato do conteúdo; evite spinner de tela cheia. Não piscar para cargas < 300ms.
- **Vazio:** explique o porquê e ofereça a próxima ação (ex.: "Nenhum alerta ativo. Tudo tranquilo por aqui." / "Ajustar filtros"). Diferencie "nada existe" de "o filtro não retornou nada".
- **Erro:** mensagem humana do que aconteceu + o que fazer (botão "Tentar novamente"). Nunca exiba stack trace ou mensagem crua da API; erros de domínio são traduzidos na borda.
- **Erro de formulário:** inline, junto ao campo, associado via `aria-describedby`; preserve o que o usuário digitou.
- Erros parciais não derrubam a página inteira — isole por seção.

## Dark mode

- Cores apenas via tokens semânticos (`--color-bg`, `--color-surface`, `--color-text`, `--color-danger`…), nunca hex no componente.
- Respeite `prefers-color-scheme` por padrão e permita override manual persistido.
- Fundo escuro não é `#000` puro nem texto `#fff` puro; use elevação por clareamento de superfície, não por sombra.
- Cores de severidade (crítico/alto/médio/baixo) têm variantes próprias por tema e mantêm contraste nos dois.
- Imagens, gráficos e ícones devem ser legíveis em ambos os temas; teste toda tela nos dois.

## Acessibilidade (WCAG 2.1 AA)

- Contraste mínimo 4.5:1 para texto normal, 3:1 para texto grande e componentes de UI.
- Nunca usar só cor para comunicar estado — combine com ícone e/ou texto (ex.: severidade).
- HTML semântico primeiro (`button`, `a`, `nav`, `main`, `table`); ARIA só quando não houver elemento nativo.
- Tudo operável por teclado, com ordem de foco lógica e `:focus-visible` sempre visível.
- Alvos de toque com no mínimo 44×44px.
- Todo input tem `label` associado; ícones-botão têm `aria-label`; imagens têm `alt` (vazio se decorativas).
- Atualizações dinâmicas relevantes (novo alerta, incidente resolvido) anunciadas via `aria-live`.
- Respeite `prefers-reduced-motion`.
