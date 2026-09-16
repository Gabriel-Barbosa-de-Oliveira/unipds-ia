import { z } from "zod";

import { createModel } from "../agents/model.ts";
import type { MemoryStore } from "./memory-store.ts";

const DISTILL_PROMPT =
  "Você é o refletor de aprendizado de um copiloto de plantão. Analise APENAS a última " +
  "mensagem da pessoa usuária (nunca o restante da conversa) e decida se ela contém um fato " +
  "durável sobre a pessoa ou sua forma de trabalhar — algo que continua válido além desta " +
  "conversa (ex.: \"eu sou o responsável pelo serviço de pagamentos\"). NUNCA trate um pedido " +
  "pontual ou uma pergunta (ex.: \"abra um incidente\", \"qual o status do serviço X?\") como " +
  "aprendizado. NUNCA inclua segredo ou credencial (senha, token, chave de API) em `fact`, " +
  "mesmo que a mensagem contenha um fato genuíno ao lado — quando o fato e o segredo forem " +
  "inseparáveis, responda hasLearning: false. Se não houver fato elegível, responda " +
  "hasLearning: false e omita `fact`.";

const learningSchema = z.object({
  hasLearning: z.boolean(),
  fact: z
    .string()
    .min(1)
    .optional()
    .describe("O fato durável já destilado, em linguagem natural. Presente só quando hasLearning é true."),
});

export type LearningVerdict = z.infer<typeof learningSchema>;

/**
 * Analisa `message` (a última mensagem da pessoa usuária) e decide se ela contém um fato
 * durável elegível para memória, via `withStructuredOutput` sobre o modelo de raciocínio — mesmo
 * padrão de `src/agents/reflection.ts#critique`.
 */
export async function distillLearning(message: string): Promise<LearningVerdict> {
  const verdict = await createModel()
    .withStructuredOutput(learningSchema)
    .invoke([
      ["system", DISTILL_PROMPT],
      ["user", message],
    ]);

  if (!verdict) {
    throw new Error(
      "Refletor de aprendizado não retornou um veredito estruturado válido (o modelo não chamou a ferramenta esperada)",
    );
  }

  return verdict;
}

export type DistillFn = (message: string) => Promise<LearningVerdict>;

/**
 * Orquestra a decisão de aprendizado: chama `distillFn` e, quando `hasLearning` e `fact` estão
 * presentes, registra o fato via `store.remember` (mesmo método já usado pela tool
 * `remember_fact`, `007-semantic-memory` — inclui a mesma deduplicação). Nunca rejeita — qualquer
 * erro de `distillFn` ou `store.remember` é absorvido e só gera um `console.error` de diagnóstico
 * (FR-006), para que quem chama possa disparar sem `await` e sem `.catch` com segurança.
 */
export async function reflectAndRemember(
  store: MemoryStore,
  userId: string,
  message: string,
  distillFn: DistillFn = distillLearning,
): Promise<void> {
  try {
    const verdict = await distillFn(message);
    if (verdict.hasLearning && verdict.fact) {
      await store.remember(userId, verdict.fact);
    }
  } catch (error) {
    console.error("Refletor de aprendizado falhou (ignorado, FR-006):", error);
  }
}
