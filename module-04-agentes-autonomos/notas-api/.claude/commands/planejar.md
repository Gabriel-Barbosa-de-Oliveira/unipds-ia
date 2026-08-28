---
description: Gera o plano técnico (COMO) a partir de uma spec existente.
---

Gera o PLANO TÉCNICO da feature indicada (o texto após o comando é numero ou o caminho)

- leia a spec.md da feature e a specs/constitution.md
- crie plan.md na mesma pasta da spec, contendo
1. Arquitetura - camadas/arquivos criados ou alterados(http/cli -> service -> store).
2. Modelo de dados - tipos e schemads zod.
3. Contratos - rotas HTTP e/ou comandos de CLI, com entrada/saida.
4. Decisões e trade-offs.
5. Estratégia de testes(node:test)

- Não implemente nada ainda. Aponte riscos e pontos que precisam de decisão humana
