import { MultiServerMCPClient } from "@langchain/mcp-adapters";

export const getMCPTools = async () => {
    const mcpClient = new MultiServerMCPClient({
        fileSystem: {
            transport: "stdio",
            command: "npx",
            args: [
                '-y',
                '@modelcontextprotocol/server-filesystem',
                process.cwd() //curent working directory
            ]
        }
    });


    return mcpClient.getTools();
}