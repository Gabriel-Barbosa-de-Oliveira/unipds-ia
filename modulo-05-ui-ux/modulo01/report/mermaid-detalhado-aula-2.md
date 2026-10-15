# COMANDO DE ENGENHARIA DE FLUXO

Atue como um Arquiteto de Soluções Sênior. Com base na análise de riscos e requisitos refinados que acabamos de gerar acima, crie o código para um Diagrama de Fluxo (Flowchart) utilizando a sintaxe **Mermaid.js**.

## Requisitos do Diagrama:
1. **Orientação:** Top-Down (`graph TD`).
2. **Cobertura:** Deve cobrir o "Caminho Feliz" (Sucesso) e TODOS os "Caminhos Infelizes" (Erros de API, Validação, Timeout, Falta de Saldo) identificados na análise anterior.
3. **Estados de Interface:** Represente claramente telas de *Loading*, *Empty State* e *Feedback de Erro*.
4. **Estilização Semântica:**
   - Use nós retangulares `[]` para Ações do Usuário ou Processos do Sistema.
   - Use losangos `{}` para Decisões de Lógica de Negócio (ex: "Tem saldo?", "API Online?").
   - **Importante:** Aplique estilos (classes) para diferenciar visualmente:
     - `classDef error fill:#f96,stroke:#333,stroke-width:2px;` (Para erros)
     - `classDef success fill:#9f6,stroke:#333,stroke-width:2px;` (Para sucesso)

## Saída Esperada:
Apenas o bloco de código Markdown (```mermaid```) pronto para renderização. Não inclua explicações em texto.

```mermaid
graph TD
    Start([Início: Usuário acessa Agendar Pix]) --> LoadContacts[/Loading: Carregando contatos/]
    LoadContacts --> ContactsAPI{API de contatos respondeu?}
    ContactsAPI -- Não --> ErrContacts[/Erro: Falha ao carregar contatos/]:::error
    ErrContacts --> RetryContacts[Usuário toca em Tentar novamente]
    RetryContacts --> LoadContacts
    ContactsAPI -- Sim --> HasContacts{Possui contatos salvos?}
    HasContacts -- Não --> EmptyContacts[/Empty State: Digitar chave Pix ou adicionar contato/]
    HasContacts -- Sim --> SelectContact[Usuário seleciona contato]
    EmptyContacts --> SelectContact
    SelectContact --> ContactBlocked{Contato bloqueado ou restrito?}
    ContactBlocked -- Sim --> ErrContactBlocked[/Erro: Contato indisponível para Pix/]:::error
    ContactBlocked -- Não --> InputValue[Usuário informa valor]
    InputValue --> ValidValue{Valor maior que zero?}
    ValidValue -- Não --> ErrValue[/Erro inline: Valor inválido/]:::error
    ErrValue --> InputValue
    ValidValue -- Sim --> DailyLimit{Dentro do limite diário de R$ 5.000,00?}
    DailyLimit -- Não --> ErrLimit[/Erro: Limite diário excedido/]:::error
    ErrLimit --> InputValue
    DailyLimit -- Sim --> SelectDate[Usuário seleciona data]
    SelectDate --> IsToday{Data é hoje ou passada?}
    IsToday -- Sim --> RedirectNormal[Redireciona para Pix normal]
    IsToday -- Não --> MaxDate{Dentro da data máxima de agendamento?}
    MaxDate -- Não --> ErrDate[/Erro: Data fora do período permitido/]:::error
    ErrDate --> SelectDate
    MaxDate -- Sim --> Confirm[Usuário confirma agendamento]
    Confirm --> Auth{Autenticação válida?}
    Auth -- Não --> ErrAuth[/Erro: Autenticação recusada/]:::error
    ErrAuth --> Confirm
    Auth -- Sim --> FraudCheck{Análise antifraude aprovada?}
    FraudCheck -- Não --> ErrFraud[/Erro: Operação bloqueada para revisão/]:::error
    FraudCheck -- Sim --> Submitting[/Loading: Enviando agendamento/]
    Submitting --> APIOnline{API de agendamento online?}
    APIOnline -- Não --> ErrAPI[/Erro: Serviço indisponível/]:::error
    ErrAPI --> RetryConfirm{Usuário tenta novamente?}
    RetryConfirm -- Sim --> Confirm
    RetryConfirm -- Não --> EndFail([Fim sem agendamento])
    APIOnline -- Sim --> Timeout{Resposta recebida no tempo limite?}
    Timeout -- Não --> ErrTimeout[/Erro: Tempo esgotado. Verificando status/]:::error
    ErrTimeout --> CheckStatus{Agendamento foi criado?}
    CheckStatus -- Sim --> ShowReceipt
    CheckStatus -- Não --> ErrNotCreated[/Erro: Nada foi cobrado. Tentar novamente/]:::error
    ErrNotCreated --> Confirm
    Timeout -- Sim --> SchedValid{Agendamento aceito pela API?}
    SchedValid -- Não --> ErrBalance{Motivo: Saldo insuficiente?}
    ErrBalance -- Sim --> ErrFunds[/Erro: Saldo insuficiente/]:::error
    ErrFunds --> EndFail
    ErrBalance -- Não --> ErrGeneric[/Erro: Falha ao agendar/]:::error
    ErrGeneric --> EndFail
    SchedValid -- Sim --> ShowReceipt[/Comprovante: Pix agendado com sucesso/]:::success
    ShowReceipt --> ListSched[Lista de agendamentos]
    ListSched --> HasSched{Possui agendamentos?}
    HasSched -- Não --> EmptySched[/Empty State: Nenhum Pix agendado/]
    HasSched -- Sim --> SelectSched[Usuário seleciona agendamento]
    SelectSched --> CanCancel{Ainda em status Agendado?}
    CanCancel -- Não --> ErrNotCancelable[/Erro: Pix já processado, não é possível cancelar/]:::error
    CanCancel -- Sim --> ConfirmCancel[Usuário confirma cancelamento]
    ConfirmCancel --> CancelAPI{API de cancelamento respondeu?}
    CancelAPI -- Não --> ErrCancel[/Erro: Falha ao cancelar. Agendamento mantido/]:::error
    ErrCancel --> ConfirmCancel
    CancelAPI -- Sim --> CancelOK[/Sucesso: Agendamento cancelado/]:::success
    CancelOK --> ListSched
    EmptySched --> EndOK([Fim])

    subgraph Execucao_na_Data [Execução no servidor na data agendada]
        ExecStart([Job na data agendada]) --> ExecBalance{Saldo suficiente?}
        ExecBalance -- Não --> ExecFail[/Falha: Saldo insuficiente. Notificar usuário/]:::error
        ExecBalance -- Sim --> ExecLimit{Dentro do limite diário na data?}
        ExecLimit -- Não --> ExecFailLimit[/Falha: Limite diário excedido. Notificar usuário/]:::error
        ExecLimit -- Sim --> ExecContact{Chave Pix e contato válidos?}
        ExecContact -- Não --> ExecFailKey[/Falha: Chave inválida. Notificar usuário/]:::error
        ExecContact -- Sim --> ExecPay[Processa Pix]
        ExecPay --> ExecDone[/Sucesso: Pix executado. Notificar usuário/]:::success
    end

    classDef error fill:#f96,stroke:#333,stroke-width:2px;
    classDef success fill:#9f6,stroke:#333,stroke-width:2px;
```