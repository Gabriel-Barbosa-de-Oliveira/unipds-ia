describe('Cadastro de Eventos', () => {
  const fields = ['name', 'address', 'capacity', 'date'];

  beforeEach(() => {
    cy.visit('/event/new');
  });

  it('cadastra um evento com dados válidos', () => {
    cy.get('input[name="name"]').type('Angular Summit');
    cy.get('input[name="address"]').type('Av. Paulista, 1000');
    cy.get('input[name="capacity"]').type('200');
    cy.get('input[name="date"]').type('2027-03-15');

    cy.get('button[type="submit"]').click();

    cy.get('.form-status--success').should(
      'contain.text',
      'Evento "Angular Summit" cadastrado com sucesso.'
    );
    fields.forEach((field) => {
      cy.get(`input[name="${field}"]`).should('have.value', '');
    });
  });

  it('exibe as mensagens de validação ao enviar o formulário vazio', () => {
    cy.get('button[type="submit"]').click();

    cy.contains('#event-name-error', 'Nome é obrigatório.');
    cy.contains('#event-address-error', 'Endereço é obrigatório.');
    cy.contains('#event-capacity-error', 'Capacidade é obrigatória.');
    cy.contains('#event-date-error', 'Data é obrigatória.');

    fields.forEach((field) => {
      cy.get(`input[name="${field}"]:invalid`).should('exist');
    });
    cy.get('.form-status--success').should('not.exist');
  });
});
