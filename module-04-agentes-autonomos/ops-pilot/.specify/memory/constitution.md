<!--
Sync Impact Report
- Version change: (unset/template) → 1.0.0
- Modified principles: n/a (initial ratification)
- Added sections:
  - Core Principles: I. Camadas Explícitas, II. Validação na Fronteira, III. Erros de Domínio,
    IV. Funções Puras, V. Teste Obrigatório, VI. Segurança por Padrão, VII. Spec Antes de Código,
    VIII. Pequeno e Reversível
  - Stack Tecnológica Obrigatória
  - Fluxo de Desenvolvimento & Quality Gates
  - Governance
- Removed sections: n/a
- Templates requiring updates: no dependent templates modified by this command (read constitution
  at runtime per Scope Guard); no other references to a prior constitution version were found.
- Follow-up TODOs:
  - TODO(RATIFICATION_DATE): original adoption date of the project's governing principles is
    unknown; using the date this constitution was formalized in Spec Kit format as the
    ratification date.
-->

# OpsPilot Constitution

## Core Principles

### I. Camadas Explícitas
Camadas padrão MVC (Model, Service, Controller). Dependências fluem em uma única direção:
http/cli → controller → service → model. O domínio (regras de negócio) NÃO realiza IO
diretamente; IO fica restrito às camadas de service e model.
Rationale: mantém o domínio testável em isolamento e evita acoplamento entre transporte
(HTTP/CLI) e regras de negócio.

### II. Validação na Fronteira
Toda entrada externa (HTTP/CLI) DEVE ser validada com zod antes de virar domínio. Dados que
cruzam a fronteira sem passar por um schema zod são tratados como bug.
Rationale: garante que o domínio opera apenas sobre dados já sanitizados e tipados, eliminando
validações defensivas espalhadas pelo código.

### III. Erros de Domínio
Falhas previsíveis DEVEM ser modeladas como classes de erro de domínio, nunca como strings ou
códigos soltos. A tradução para status HTTP/saída de CLI acontece somente na borda (controller),
nunca dentro do domínio ou do service.
Rationale: mantém o domínio livre de detalhes de transporte e centraliza a política de resposta
a erros em um único ponto.

### IV. Funções Puras
Lógica de domínio é escrita preferencialmente como funções puras (mesma entrada, mesma saída,
sem efeitos colaterais). Efeitos colaterais (rede, banco, filesystem, tempo, aleatoriedade)
DEVEM ficar isolados nas camadas de service/model.
Rationale: funções puras são triviais de testar sem mocks pesados e reduzem bugs de estado
oculto.

### V. Teste Obrigatório (NON-NEGOTIABLE)
Nenhuma lógica nova entra sem teste (`node:test` via `tsx`). `npm run typecheck` e `npm test`
DEVEM estar sempre verdes antes de qualquer merge; o pre-commit hook que executa ambos não pode
ser contornado (`--no-verify`) sem aprovação explícita.
Rationale: TypeScript `strict` mais testes automatizados são a rede de segurança do projeto —
sem eles, alterações em um agente LangGraph são difíceis de validar manualmente.

### VI. Segurança por Padrão
Segredos NUNCA são commitados no repositório. O arquivo `.env` NUNCA é lido pelo agente ou por
ferramentas automatizadas. Ações destrutivas (ex.: comandos que alteram infraestrutura de
produção, dados de incidentes reais) passam por guardrails explícitos (deny list + pre-commit),
não pela confiança no julgamento do modelo em tempo de execução.
Rationale: OpsPilot opera sobre alertas e incidentes de produção reais; erros de automação têm
custo alto e devem ser barrados estruturalmente, não apenas por prompt.

### VII. Spec Antes de Código
Mudanças relevantes seguem o fluxo do Spec Kit: `/speckit.specify` → `/speckit.plan` →
`/speckit.tasks` → `/speckit.implement`, com revisão humana entre as fases. Specs são
versionadas junto ao código.
Rationale: força alinhamento sobre o "o quê" e o "por quê" antes de gastar esforço de
implementação, e deixa rastro auditável de decisões.

### VIII. Pequeno e Reversível
Cada tarefa cabe em um commit. Mudanças grandes são decompostas em incrementos pequenos e
revertíveis independentemente.
Rationale: reduz o raio de impacto de um erro e facilita review, bisect e rollback.

## Stack Tecnológica Obrigatória

- Node 24 LTS
- TypeScript ESM, `strict`
- zod na fronteira (HTTP/CLI)
- Testes com `node:test` via `tsx`
- Express como servidor HTTP
- Sequelize + MySQL como banco
- LangChain/LangGraph sobre OpenRouter para o agente

Mudar qualquer item desta stack é uma decisão arquitetural e exige amendment desta constitution.

## Fluxo de Desenvolvimento & Quality Gates

- `npm run dev` inicia a API (`src/index.ts`); `npm run arena` e `npm run bench` rodam os
  scripts de avaliação do agente (`src/arena.ts`, `src/bench.ts`).
- `npm test` (`node --import tsx --test`) e `npm run typecheck` (`tsc --noEmit`) DEVEM passar
  antes de qualquer commit; o pre-commit hook automatiza essa checagem.
- Specs, planos e tarefas geradas pelo Spec Kit vivem em `specs/` e são versionadas como
  qualquer outro artefato do repositório.

## Governance

Esta constitution tem precedência sobre qualquer outra prática, convenção ou preferência
individual dentro do projeto. Em caso de conflito, a constitution vence.

Amendments (mudança de princípio, adição/remoção de seção, ou mudança de stack obrigatória)
exigem: (1) registro explícito do racional da mudança, (2) atualização deste documento com o
Sync Impact Report correspondente, e (3) bump de versão semântico:
- MAJOR: remoção ou redefinição incompatível de um princípio.
- MINOR: novo princípio ou seção adicionada, ou expansão material de uma diretriz existente.
- PATCH: clarificação, correção de texto, ou refinamento não-semântico.

Toda PR/review DEVE verificar conformidade com os princípios acima. Complexidade que viole um
princípio (ex.: IO dentro do domínio, entrada não validada, ausência de teste) precisa ser
justificada explicitamente na PR ou corrigida antes do merge. Uso diário do dia a dia é guiado
por `CLAUDE.md`, que deve permanecer consistente com esta constitution.

**Version**: 1.0.0 | **Ratified**: 2026-09-01 | **Last Amended**: 2026-09-01
