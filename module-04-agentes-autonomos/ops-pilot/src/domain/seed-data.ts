import type { Alert, OpsState, Service } from "./ops-store.ts";

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

/**
 * Constrói o dataset canônico (5 serviços, 6 alertas: 3 firing, 3 resolved, sem incidentes).
 * Cada chamada retorna um novo objeto — nenhum estado compartilhado é mutado entre chamadas.
 */
export function buildSeedState(): OpsState {
  return {
    services: SERVICES.map((service) => ({ ...service })),
    alerts: ALERTS.map((alert) => ({ ...alert })),
    incidents: [],
  };
}
