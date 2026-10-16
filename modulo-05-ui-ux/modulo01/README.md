O GitHub bloqueou o acesso automatizado à página. Vou tentar clonar o repositório pelo shell.

```markdown
# Discovery & Refinement: Pix Agendado com LLMs

Pipeline de Discovery e Refinamento de requisitos para a feature de **Pix Agendado**, desenvolvido no módulo 05 (UX e UI) da pós-graduação em Engenharia de IA Aplicada. O projeto utiliza LLMs (Google AI Studio / Gemini) orientados por engenharia de prompt para transformar um ticket bruto em análise de risco, mapeamento de estados de interface, diagrama de fluxo, catálogo de mensagens de UX, sanitização de feedbacks de usuários e backlog priorizado.

## Arquitetura do Pipeline

O fluxo é composto por dois ramos que partem de fontes distintas e convergem para artefatos de engenharia.

**Ramo 1: Requisito para Implementação**

1. **Entrada:** `docs/briefing-bruto.md` contém o ticket original, com descrição e regras de negócio (limite diário de R$ 5.000,00, bloqueio de agendamento para o mesmo dia e cancelamento posterior).
2. **Análise de requisito:** `prompts-e-skills/analista-requisitos.md` é aplicado ao briefing. O resultado está em `docs/refinement/refinamento.md` e contém análise de risco, mapeamento de estados de UI (loading, vazio e erro), cenários ocultos e regras de negócio conflitantes.
3. **Diagrama de fluxo:** `report/mermaid-detalhado-aula-2.md` contém o comando de engenharia e o diagrama Mermaid.js (`graph TD`) gerado a partir da análise. Cobre o caminho feliz e os caminhos de erro de API, validação, timeout e falta de saldo, além da execução no servidor na data agendada.
4. **Mensagens de UX:** `prompts-e-skills/ux-writer.md` define o style guide (tom sem culpa, resolutivo, consistência técnica) e o schema JSON. A tarefa de extração está em `report/ux-writer-aula-3.md`, e o resultado em `report/pt-Br.json` tem title, message e action_label para cada cenário de erro, validação e sucesso.

**Ramo 2: Feedback do Usuário para Backlog**

1. **Entrada:** `data/raw-feedbacks.json` contém feedbacks brutos de tickets de suporte, com dados pessoais e ruído (mensagens de bot, testes internos e reclamações sobre atendimento físico).
2. **Sanitização (LGPD):** `prompts-e-skills/data-sanitizer.md` remove PII (CPF, e-mail, telefone e nomes de terceiros) substituindo por `[REDACTED]`, descarta ruído e substitui autores por IDs anonimizados. O resultado está em `data/sanitized-feedback.json`.
3. **Priorização:** `prompts-e-skills/product-manager.md` classifica cada feedback em categoria (`BUG_CRITICO`, `UX_UI_IMPROVEMENT` ou `NEW_FEATURE`) e severidade (`ALTA`, `MEDIA` ou `BAIXA`), com ação proposta. O resultado está em `data/backlog.json`.

## Como utilizar

1. Clone o repositório:

   ```bash
   git clone https://github.com/Gabriel-Barbosa-de-Oliveira/unipds-ia.git
   cd unipds-ia/modulo-05-ui-ux/modulo01
   ```

2. Acesse a pasta `prompts-e-skills/`, que contém os system prompts em formato Markdown.
3. Para cada etapa, copie o conteúdo do prompt correspondente como instrução de sistema e envie o insumo da etapa como mensagem do usuário. Exemplos:
   - `analista-requisitos.md` com o conteúdo de `docs/briefing-bruto.md`.
   - `data-sanitizer.md` com o conteúdo de `data/raw-feedbacks.json`.
   - `product-manager.md` com o conteúdo de `data/sanitized-feedback.json`.
4. Execute no seu motor de LLM preferido. Os prompts foram testados com Google Gemini via AI Studio, mas são agnósticos de provedor.
5. Compare sua saída com os artefatos versionados em `docs/`, `data/` e `report/`.

**Atenção:** o arquivo `data/raw-feedbacks.json` contém dados pessoais fictícios e não deve ser enviado a serviços externos sem sanitização prévia. Use sempre o pipeline de sanitização antes de processar feedbacks reais.

## Estrutura do Diretório

```
modulo01/
├── docs/
│   ├── briefing-bruto.md          # Ticket original
│   └── refinement/
│       └── refinamento.md         # Análise de risco, estados e cenários ocultos
├── data/
│   ├── raw-feedbacks.json         # Feedbacks brutos (contém PII, não versionar em produção)
│   ├── sanitized-feedback.json    # Feedbacks anonimizados
│   └── backlog.json               # Backlog priorizado
├── prompts-e-skills/
│   ├── analista-requisitos.md     # Análise de requisitos
│   ├── data-sanitizer.md          # Sanitização LGPD
│   ├── product-manager.md         # Classificação e priorização
│   ├── ux-writer.md               # Padrão de mensagens de UX
│   └── readme-generator.md        # Geração deste README
└── report/
    ├── mermaid-detalhado-aula-2.md  # Comando e diagrama de fluxo
    ├── ux-writer-aula-3.md          # Tarefa de extração de mensagens
    └── pt-Br.json                   # Catálogo de mensagens em pt-BR
```

## Stack Tecnológico

- **Engenharia de Prompt:** prompts com papel, objetivo, regras, formato de saída e schema explícito.
- **Google Gemini (AI Studio):** motor de LLM utilizado na execução dos prompts.
- **Markdown:** documentação de requisitos, análises e prompts.
- **JSON:** formato de saída estruturado para mensagens, feedbacks sanitizados e backlog.
- **Mermaid.js:** diagramas de fluxo (`graph TD`) com classes semânticas de erro e sucesso.
- **Data Sanitization (LGPD):** remoção de PII, descarte de ruído e anonimização de autores antes da análise de produto.
```

Observações sobre o README:

- O prompt de geração pede a pasta `/prompts`, mas no repositório ela se chama `prompts-e-skills/`. Usei o nome real.
- O prompt de geração aponta para outro repositório (`unipds-engenharia-de-ia-aplicada/ferramentas-de-IA-para-UX-UI`). Usei o repositório que você informou.
- O README foi salvo em `/home/claude/README.md`.