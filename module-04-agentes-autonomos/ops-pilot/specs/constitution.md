# Constitution - OpsPilot

Princípios não-negociáveis que toda spec, plano, tarefa e código seguem.

1. Camadas explícitas. MVC padrão (Model, Service, Controller). Dependências fluem http/cli -> controller -> service -> model. Domínio não faz IO direto.

2. Validações na fronteira. Toda entrada externa (HTTP/CLI) é validada com zod antes de virar domínio.

3. Erros são de domínio. Falhas previsíveis viram classes de erro, traduzidas em status/saída na borda.

4. Funções puras por padrão. Lógica de domínio evita efeitos colaterais; IO fica isolado nas camadas de service/model.

5. Teste é parte da tarefa. Nenhuma lógica nova entra sem teste. typecheck e test sempre verdes.

6. Segurança por padrão. Sem segredos no repo, nunca ler `.env`. Ações destrutivas passam por guardrails (deny list + pre-commit), não pela confiança no modelo.

7. Spec antes de código. Mudanças relevantes passam por `/speckit.specify` -> `/speckit.plan` -> `/speckit.tasks` -> `/speckit.implement`, com revisão humana entre as fases.

8. Pequeno e reversível. Cada tarefa cabe em um commit.

## Stack Obrigatória

Node 24 LTS, TypeScript ESM `strict`, zod, `node:test` via `tsx`, Express, Sequelize + MySQL, LangChain/LangGraph sobre OpenRouter.
