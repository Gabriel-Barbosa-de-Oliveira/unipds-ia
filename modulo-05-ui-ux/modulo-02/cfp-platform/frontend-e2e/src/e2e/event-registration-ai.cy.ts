describe("Cadastro de Eventos - AI Deiven Testing", () => {
    it("Deve executar o fluxo de cadastro e validar o sucesso de forma semantica", () => {
        cy.visit("/event/new");

        //Multiplos passos em uma unica chamada de array para otimizar o LLM 
        cy.prompt([
            'Type "Auditorio Oracle" in the event name field',
            'Type "Av. Dr. Chucri Zaidan, SP" in the addressfield',
            'Type "500" in the capacity field',
            'Type "2026-12-31" in the event date field',
            'Click the button that submits or saves the event',
        ])

        // Asserção baseada na intenção visual 
        cy.prompt(['Verify that a success message is visible']);
    })
});