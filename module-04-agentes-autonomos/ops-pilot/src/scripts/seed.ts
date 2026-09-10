import { seedCanonicalScenario, SqliteOpsStore } from "../store/sqlite-ops-store.ts";

const DEFAULT_DB_PATH = "./data/opspilot.db";

/**
 * Semeia/restaura o `SqliteOpsStore` (`OPSPILOT_DB`, default `./data/opspilot.db`) com o
 * cenário canônico (5 serviços, 6 alertas: 3 firing, 3 resolved, 3 runbooks, 0 incidentes).
 * Idempotente: reexecutar produz o mesmo estado (SC-005).
 */
function seed(): void {
  const dbPath = process.env.OPSPILOT_DB ?? DEFAULT_DB_PATH;
  const store = new SqliteOpsStore(dbPath);
  seedCanonicalScenario(store);
  console.log(`Seed concluído em ${dbPath}: 5 serviços, 6 alertas, 3 runbooks (0 incidentes).`);
}

try {
  seed();
} catch (error) {
  console.error("Falha ao semear o dataset:", error);
  process.exitCode = 1;
}
