import { DataTypes, Model, type InferAttributes, type InferCreationAttributes } from "sequelize";

import { getSequelizeConnection } from "./connection.ts";

export class ServiceModel extends Model<InferAttributes<ServiceModel>, InferCreationAttributes<ServiceModel>> {
  declare id: string;
  declare name: string;
}

ServiceModel.init(
  {
    id: { type: DataTypes.STRING, primaryKey: true },
    name: { type: DataTypes.STRING, allowNull: false, unique: true },
  },
  {
    sequelize: getSequelizeConnection(),
    modelName: "Service",
    tableName: "services",
    timestamps: false,
  },
);
