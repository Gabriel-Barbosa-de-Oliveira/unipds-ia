import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { decrypt, encrypt } from "./service.ts";
import { server } from "./mcp.ts";

async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport)
    //não tem console log em server mcp
    console.error('Encrypt MCP Server running on stdio')
}

main().catch((error) => {
    console.error("Fatal error in main():", error);
    process.exit(1);
});