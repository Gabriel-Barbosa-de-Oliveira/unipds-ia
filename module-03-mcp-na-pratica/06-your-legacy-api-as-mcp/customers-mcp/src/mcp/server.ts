import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CustomerHttpClient } from "../infrastructure/customerHttpClient.ts";
import { registerListCustomersTool } from "./tools/listCustomers.ts";
import { CustomerService } from "../application/customerService.ts";
import { registerApiInfoResource } from "./resources/apiInfo.ts";

const BASE_URL = "http://localhost:9999/v1";
const service = new CustomerService(BASE_URL)

export const server = new McpServer({
    name: "@erickwendel/ew-customers-mcp",
    version: "0.0.1",
});

registerListCustomersTool(server, service)
registerApiInfoResource(server, BASE_URL)