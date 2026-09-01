# Spec 002 — Persistência de Tarefas em Arquivo JSON

## 1. Contexto/Problema

Hoje (spec 001) tanto a CLI quanto o servidor HTTP usam uma store em memória, existente apenas durante a vida do processo. Isso significa que: (a) as tarefas criadas numa execução da CLI desaparecem assim que o processo termina, e (b) CLI e servidor HTTP não compartilham nenhum estado entre si, mesmo rodando ao mesmo tempo. É preciso persistir as tarefas em um arquivo `.json` para que sobrevivam entre execuções da CLI e sejam compartilhadas entre CLI e servidor HTTP.

## 2. User Stories

- Como usuário da CLI, quero que as tarefas criadas numa execução ainda existam na próxima execução, para não perder meu trabalho ao fechar o terminal.
- Como usuário, quero que uma tarefa criada pela CLI apareça também no servidor HTTP (e vice-versa), para ter uma única fonte de verdade sobre minhas tarefas.
- Como usuário, quero que o sistema funcione normalmente na primeira execução, mesmo sem o arquivo de dados existir ainda.
- Como usuário, quero ser avisado claramente se o arquivo de dados estiver corrompido ou em formato inesperado, para não perder dados silenciosamente.

## 3. Requisitos Funcionais

- **RF-1**: O sistema deve persistir as tarefas em um arquivo `.json`, em um caminho fixo dentro do projeto.
- **RF-2**: Tanto a CLI quanto o servidor HTTP devem ler e escrever nesse mesmo arquivo, de forma que as tarefas visíveis por uma interface sejam as mesmas visíveis pela outra.
- **RF-3**: Ao iniciar (CLI ou HTTP) e o arquivo de dados não existir, o sistema deve tratar o estado inicial como uma lista vazia de tarefas e criar o arquivo.
- **RF-4**: Toda operação que cria, conclui ou remove uma tarefa deve persistir o novo estado no arquivo antes de confirmar sucesso ao chamador (CLI ou HTTP).
- **RF-5**: Ao ler um arquivo existente cujo conteúdo não seja JSON válido, ou que não corresponda ao formato esperado de lista de tarefas, o sistema deve recusar a operação com uma mensagem de erro clara, sem sobrescrever ou apagar o conteúdo do arquivo.
- **RF-6**: O formato das tarefas gravadas no arquivo deve ser compatível com o modelo de domínio já existente (id, title, status).

## 4. Critérios de Aceite (EARS)

- Quando a CLI ou o servidor HTTP iniciar e o arquivo de dados não existir, o sistema deve criar o arquivo com uma lista vazia de tarefas.
- Quando uma tarefa for criada, o sistema deve gravar o novo estado no arquivo antes de confirmar sucesso.
- Quando uma tarefa for concluída, o sistema deve gravar o novo estado no arquivo antes de confirmar sucesso.
- Quando uma tarefa for removida, o sistema deve gravar o novo estado no arquivo antes de confirmar sucesso.
- Quando a CLI for executada após uma execução anterior (da CLI ou do servidor) ter criado ou alterado tarefas, o sistema deve listar essas tarefas com o estado mais recente.
- Quando uma tarefa for criada ou alterada via CLI, uma consulta subsequente ao servidor HTTP deve refletir essa mudança.
- Quando uma tarefa for criada ou alterada via HTTP, uma execução subsequente da CLI deve refletir essa mudança.
- Quando o arquivo de dados existir mas contiver JSON inválido ou fora do formato esperado, o sistema deve recusar a operação com uma mensagem de erro clara, sem apagar ou sobrescrever o conteúdo do arquivo.

## 5. Fora de Escopo

- Locking de arquivo ou qualquer proteção contra escrita concorrente simultânea (CLI e HTTP, ou duas CLIs, escrevendo no mesmo instante) — o comportamento sob concorrência real fica indefinido nesta fase.
- Migração automática de um formato de arquivo antigo para um novo.
- Configuração do caminho do arquivo via variável de ambiente ou flag de linha de comando.
- Backup, versionamento ou histórico de mudanças do arquivo.
- Múltiplos arquivos/fontes de dados distintos por ambiente (dev/test/prod).

## 6. Questões em Aberto

Nenhuma — ambiguidades iniciais foram resolvidas com o usuário: persistência compartilhada entre CLI e HTTP (mesmo arquivo), caminho fixo dentro do projeto, e comportamento definido para arquivo ausente (cria vazio) vs. corrompido (erro claro, sem apagar dados).
