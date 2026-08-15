import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { getMongoDBTool } from "../tools/mongodbTool.ts";
import { getCSVTOJSONTool } from "../tools/csvToJSONTool.ts";
import { getFSTool } from "../tools/fsTool.ts";

let mcpClient: MultiServerMCPClient | undefined;

export const getMCPTools = async () => {
  mcpClient = new MultiServerMCPClient({
    mcpServers: {
      ...getMongoDBTool(),
      ...getFSTool()
    },
    onMessage: (log, source) => {
      console.log(`${source.server} ${log.data}`)
    }
  })

  const mcpTools = await mcpClient.getTools()

  return [
    ...mcpTools,
    getCSVTOJSONTool()
  ];
};

export const closeMCPTools = async () => {
  if (mcpClient) {
    await mcpClient.close();
    mcpClient = undefined;
  }
};
