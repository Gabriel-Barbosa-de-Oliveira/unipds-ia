#Constitution - Notas API 

Principios não-negociaveis que toda spec, plano, tarefa e codigo seguem. 

1. Camadas Explicictas.Dependencias fluem http/cli -> service -> store. Dominio não faz IO.

2. Validações na fronteira. Toda entrada externa é validada com zod antes de virar dominio. 

3. Erros são de dominio. Falhas previsiveis viram classes de erro, traduzidas em status/saída na borda 

4. Teste é parte da tarefa. Nenhuma logica nova entra sem teste. typecheck e test sempre verdes

5- Segurança por padrão. Sem segredos no repo. Ações destruitivas passam por guardrails (deny list + pre-commit), não pela confiaça no modelo.

6. Spec antes de codigo. Mudanças relevantes passam por spec -> plan -> task -> implement, com revisão humana entre as fases. 

7. Pequeno e recersível. Cada tarefa cabe em um commit 

## Stack Obrigatoria 

Node 22, Typescript ESM strict, zod, node: test via tsx, node:http
