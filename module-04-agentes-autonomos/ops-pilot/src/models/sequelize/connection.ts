import { Sequelize } from "sequelize";

let instance: Sequelize | undefined;

/**
 * Fábrica única da conexão Sequelize/MySQL. Lê as credenciais de `process.env` em runtime
 * (nunca do arquivo `.env` diretamente) — usada apenas pelo adaptador de persistência
 * (`src/services/ops-store.sequelize.ts`) e pelo script de seed, nunca pelo domínio.
 */
export function getSequelizeConnection(): Sequelize {
  if (instance) {
    return instance;
  }

  instance = new Sequelize({
    dialect: "mysql",
    host: process.env.MYSQL_HOST ?? "localhost",
    port: process.env.MYSQL_PORT ? Number(process.env.MYSQL_PORT) : 3306,
    username: process.env.MYSQL_USER,
    password: process.env.MYSQL_PASSWORD,
    database: process.env.MYSQL_DATABASE,
    logging: false,
  });

  return instance;
}
