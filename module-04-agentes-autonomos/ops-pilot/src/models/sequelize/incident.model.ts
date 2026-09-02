import { DataTypes, Model, type InferAttributes, type InferCreationAttributes } from "sequelize";

import { getSequelizeConnection } from "./connection.ts";

export class IncidentModel extends Model<
  InferAttributes<IncidentModel>,
  InferCreationAttributes<IncidentModel>
> {
  declare id: string;
  declare title: string;
  declare serviceId: string;
  declare severity: "low" | "medium" | "high" | "critical";
  declare status: "open" | "resolved";
  declare createdAt: string;
  declare resolvedAt: string | null;
}

IncidentModel.init(
  {
    id: { type: DataTypes.STRING, primaryKey: true },
    title: { type: DataTypes.STRING, allowNull: false },
    serviceId: { type: DataTypes.STRING, allowNull: false },
    severity: {
      type: DataTypes.ENUM("low", "medium", "high", "critical"),
      allowNull: false,
    },
    status: { type: DataTypes.ENUM("open", "resolved"), allowNull: false },
    createdAt: { type: DataTypes.DATE, allowNull: false },
    resolvedAt: { type: DataTypes.DATE, allowNull: true },
  },
  {
    sequelize: getSequelizeConnection(),
    modelName: "Incident",
    tableName: "incidents",
    timestamps: false,
  },
);
