import type { Alert, OpsState, Runbook, Service } from "./ops-store.ts";

const SEED_TIMESTAMP = "2026-01-01T00:00:00.000Z";

const SERVICES: readonly Service[] = [
  { id: "svc-checkout-api", name: "checkout-api" },
  { id: "svc-payments-api", name: "payments-api" },
  { id: "svc-inventory-service", name: "inventory-service" },
  { id: "svc-notifications-service", name: "notifications-service" },
  { id: "svc-auth-service", name: "auth-service" },
];

const ALERTS: readonly Alert[] = [
  {
    id: "alert-1",
    serviceId: "svc-checkout-api",
    title: "Latência elevada no checkout",
    status: "firing",
    createdAt: SEED_TIMESTAMP,
  },
  {
    id: "alert-2",
    serviceId: "svc-payments-api",
    title: "Erro 500 em processamento de pagamento",
    status: "firing",
    createdAt: SEED_TIMESTAMP,
  },
  {
    id: "alert-3",
    serviceId: "svc-inventory-service",
    title: "Fila de sincronização atrasada",
    status: "firing",
    createdAt: SEED_TIMESTAMP,
  },
  {
    id: "alert-4",
    serviceId: "svc-notifications-service",
    title: "Falha no envio de e-mails",
    status: "resolved",
    createdAt: SEED_TIMESTAMP,
  },
  {
    id: "alert-5",
    serviceId: "svc-auth-service",
    title: "Certificado TLS expirado",
    status: "resolved",
    createdAt: SEED_TIMESTAMP,
  },
  {
    id: "alert-6",
    serviceId: "svc-checkout-api",
    title: "CPU acima de 90%",
    status: "resolved",
    createdAt: SEED_TIMESTAMP,
  },
];

const RUNBOOKS: readonly Runbook[] = [
  {
    id: "runbook-checkout-api",
    serviceId: "svc-checkout-api",
    content:
      "1. Verifique o dashboard de latência do checkout-api.\n" +
      "2. Confira se o serviço de pagamentos está saudável (dependência direta).\n" +
      "3. Se a latência persistir, escale horizontalmente o checkout-api.\n" +
      "4. Comunique o time de produto se o checkout ficar indisponível por mais de 5 minutos.",
  },
  {
    id: "runbook-payments-api",
    serviceId: "svc-payments-api",
    content:
      "1. Verifique o status do provedor de pagamentos externo.\n" +
      "2. Confira os logs de erro 500 mais recentes do payments-api.\n" +
      "3. Se o provedor externo estiver saudável, reinicie o payments-api.\n" +
      "4. Abra um incidente de severidade alta se pagamentos ficarem indisponíveis.",
  },
  {
    id: "runbook-auth-service",
    serviceId: "svc-auth-service",
    content:
      "1. Verifique a validade do certificado TLS do auth-service.\n" +
      "2. Renove o certificado se estiver expirado ou próximo do vencimento.\n" +
      "3. Reinicie o auth-service após renovar o certificado.\n" +
      "4. Confirme que outros serviços voltam a autenticar normalmente.",
  },
];

/**
 * Constrói o dataset canônico ("Mercadinho": 5 serviços, 6 alertas — 3 firing, 3 resolved —,
 * 3 runbooks, sem incidentes). Cada chamada retorna um novo objeto — nenhum estado compartilhado
 * é mutado entre chamadas.
 */
export function buildSeedState(): OpsState {
  return {
    services: SERVICES.map((service) => ({ ...service })),
    alerts: ALERTS.map((alert) => ({ ...alert })),
    incidents: [],
    runbooks: RUNBOOKS.map((runbook) => ({ ...runbook })),
  };
}
