import { describe, it, after, before } from "node:test"
import assert from "node:assert"
import { createTestClient } from "./helpers.ts"
import { Client } from "@modelcontextprotocol/sdk/client"
import { type Customer } from "../src/domain/customer.ts"

type CustomersResult = { structuredContent: { customers: Customer[] } }

describe('Customer MCP Suite', () => {
    let client: Client
    before(async () => {
        client = await createTestClient();
    })

    after(async () => {
        await client.close()
    })

    it("should list all customers", async () => {
        const result = await client.callTool({
            name: "list_customers",
            arguments: {}
        }) as unknown as CustomersResult

        assert.ok(
            Array.isArray(result.structuredContent.customers),
            "Should return an array of customers"
        )
    })
})