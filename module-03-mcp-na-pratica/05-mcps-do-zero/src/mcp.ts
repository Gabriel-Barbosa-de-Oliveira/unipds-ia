import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from 'zod/v3';
import { encrypt, decrypt } from "./service.ts";

export const server = new McpServer({
    name: "@gabrielbarbosa/cyphersuite-mcp",
    version: "0.0.1"
})

server.registerTool(
    'encrypt_message',
    {
        description: 'Encrypt a message',
        inputSchema: {
            message: z.string().describe("The message to encrypt"),
            encryptionKey: z.string().describe(
                "Any passphrase to use for encryption — the server derives a strong key from it automatically"
            )
        },
        outputSchema: {
            encryptedMessage: z.string().describe(
                "The encrypted message (format: iv:ciphertext)"
            )
        }
    },
    async ({ message, encryptionKey }) => {
        try {
            const encryptedMessage = encrypt(message, encryptionKey)
            return {
                content: [{ type: "text", text: encryptedMessage }],
                structuredContent: { encryptedMessage }
            }
        } catch (error) {
            return {
                isError: true,
                content: [{
                    type: 'text',
                    text: `Failed to encrypt message! Check if the message and encryption key are correct. Error details: ${error instanceof Error ? error.message : String(error)}`
                }]
            }
        }
    }
) 