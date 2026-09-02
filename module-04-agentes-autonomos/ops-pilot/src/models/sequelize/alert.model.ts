import { DataTypes, Model, type InferAttributes, type InferCreationAttributes } from "sequelize";

import { getSequelizeConnection } from "./connection.ts";

export class AlertModel extends Model<InferAttributes<AlertModel>, InferCreationAttributes<AlertModel>> {
  declare id: string;
  declare serviceId: string;
  declare title: string;
  declare status: "firing" | "resolved";
  declare createdAt: string;
}

AlertModel.init(
  {
    id: { type: DataTypes.STRING, primaryKey: true },
    serviceId: { type: DataTypes.STRING, allowNull: false },
    title: { type: DataTypes.STRING, allowNull: false },
    status: { type: DataTypes.ENUM("firing", "resolved"), allowNull: false },
    createdAt: { type: DataTypes.DATE, allowNull: false },
  },
  {
    sequelize: getSequelizeConnection(),
    modelName: "Alert",
    tableName: "alerts",
    timestamps: false,
  },
);
