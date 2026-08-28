# Spec 001 — Gerenciamento de Tarefas

## 1. Contexto/Problema

Hoje não existe nenhuma forma de gerenciar tarefas no sistema. É preciso permitir que um usuário crie tarefas com um título, acompanhe quais estão pendentes ou concluídas, marque-as como concluídas e as remova quando não forem mais necessárias. Essa gestão deve estar disponível tanto via API HTTP quanto via CLI, reutilizando as mesmas regras de negócio nas duas interfaces.

## 2. User Stories

- Como usuário, quero criar uma tarefa informando um título, para registrar algo que preciso fazer.
- Como usuário, quero listar todas as tarefas, para ter uma visão geral do que existe.
- Como usuário, quero listar apenas as tarefas em aberto, para saber o que ainda falta fazer.
- Como usuário, quero listar apenas as tarefas concluídas, para revisar o que já foi feito.
- Como usuário, quero concluir uma tarefa, para sinalizar que ela foi finalizada.
- Como usuário, quero remover uma tarefa, para eliminar algo que não é mais relevante.
- Como usuário, quero realizar todas essas ações tanto pela API HTTP quanto pela CLI, para escolher a interface mais conveniente em cada situação.

## 3. Requisitos Funcionais

- **RF-1**: O sistema deve permitir criar uma tarefa informando um título obrigatório.
- **RF-2**: Toda tarefa criada deve receber um identificador único (UUID) e status inicial "open".
- **RF-3**: O sistema deve permitir listar todas as tarefas, sem filtro de status.
- **RF-4**: O sistema deve permitir listar apenas tarefas com status "open".
- **RF-5**: O sistema deve permitir listar apenas tarefas com status "done".
- **RF-6**: O sistema deve permitir concluir uma tarefa existente, alterando seu status para "done".
- **RF-7**: O sistema deve permitir remover uma tarefa existente definitivamente.
- **RF-8**: O sistema deve rejeitar a criação de tarefa quando o título estiver ausente, vazio ou não for uma string.
- **RF-9**: O sistema deve sinalizar erro de domínio dedicado (tarefa não encontrada) ao tentar concluir ou remover uma tarefa com id inexistente.
- **RF-10**: Todas as operações (criar, listar, concluir, remover) devem estar disponíveis via HTTP e via CLI, com o mesmo comportamento de negócio nas duas interfaces.

## 4. Critérios de Aceite (EARS)

- Quando o usuário criar uma tarefa com um título válido, o sistema deve armazená-la com status "open" e retornar seus dados, incluindo o id gerado.
- Quando o usuário tentar criar uma tarefa sem título ou com título vazio, o sistema deve rejeitar a operação com um erro de validação, sem criar a tarefa.
- Quando o usuário solicitar a listagem "all", o sistema deve retornar todas as tarefas, independentemente do status.
- Quando o usuário solicitar a listagem "open", o sistema deve retornar apenas as tarefas com status "open".
- Quando o usuário solicitar a listagem "done", o sistema deve retornar apenas as tarefas com status "done".
- Quando o usuário concluir uma tarefa existente com status "open", o sistema deve alterar seu status para "done".
- Quando o usuário tentar concluir uma tarefa com id inexistente, o sistema deve sinalizar erro de tarefa não encontrada, sem alterar nenhum dado.
- Quando o usuário remover uma tarefa existente, o sistema deve excluí-la, de modo que ela deixe de aparecer em qualquer listagem.
- Quando o usuário tentar remover uma tarefa com id inexistente, o sistema deve sinalizar erro de tarefa não encontrada, sem alterar nenhum dado.
- Quando uma tarefa já concluída for concluída novamente, o sistema deve manter seu status como "done" sem erro.

## 5. Fora de Escopo

- Persistência duradoura (banco de dados, arquivo em disco): a store é in-memory.
- Edição de título ou de outros campos de uma tarefa existente.
- Descrição, prazos, prioridade, tags ou qualquer campo além de título, id e status.
- Autenticação, autorização ou multiusuário.
- Ordenação, paginação ou busca textual nas listagens.
- Concorrência/transações entre múltiplos processos.

## 6. Questões em Aberto

Nenhuma no momento — ambiguidades iniciais (campos da tarefa, comportamento de not-found, formato do id) foram resolvidas com o usuário: tarefa possui apenas título/id/status, erro de domínio dedicado para not-found, id em UUID.
