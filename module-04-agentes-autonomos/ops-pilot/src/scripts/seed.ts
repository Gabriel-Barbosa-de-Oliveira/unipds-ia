import { buildSeedState } from "../domain/seed-data.ts";
import { DEFAULT_DATA_FILE, writeOpsStateFile } from "../services/ops-store.memory.ts";

/**
 * Semeia/restaura o arquivo JSON local (usado como base de dados enquanto a validação desta
 * feature não depende de um MySQL real — ver `src/services/ops-store.memory.ts`) para o dataset
 * canônico (5 serviços, 6 alertas: 3 firing, 3 resolved, sem incidentes). Idempotente:
 * reexecutar produz o mesmo estado (SC-006). O adaptador Sequelize/MySQL (`ops-store.sequelize.ts`)
 * permanece disponível para quando um banco real for necessário.
 */
function seed(): void {
  const state = buildSeedState();
  writeOpsStateFile(DEFAULT_DATA_FILE, state);
  console.log(
    `Seed concluído em ${DEFAULT_DATA_FILE}: ${state.services.length} serviços, ${state.alerts.length} alertas (0 incidentes).`,
  );
}

try {
  seed();
} catch (error) {
  console.error("Falha ao semear o dataset:", error);
  process.exitCode = 1;
}
