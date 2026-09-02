import { ChatOpenAI } from "@langchain/openai";

const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

/**
 * Fábrica única do modelo de raciocínio (OpenRouter, temperature 0). Nenhuma estratégia
 * instancia um cliente de modelo diretamente — todas usam `createModel()`.
 */
export function createModel(): ChatOpenAI {
  const apiKey = process.env.OPENROUTER_API_KEY;
  const model = process.env.OPENROUTER_MODEL;

  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY não configurada");
  }
  if (!model) {
    throw new Error("OPENROUTER_MODEL não configurada");
  }

  return new ChatOpenAI({
    model,
    apiKey,
    configuration: { baseURL: OPENROUTER_BASE_URL },
    temperature: 0,
  });
}
