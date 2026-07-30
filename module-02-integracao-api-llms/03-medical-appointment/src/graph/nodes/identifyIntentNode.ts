import { getSystemPrompt, getUserPromptTemplate, IntentSchema } from '../../prompts/v1/identifyIntent.ts';
import { professionals } from '../../services/appointmentService.ts';
import { OpenRouterService } from '../../services/openRouterService.ts';
import type { GraphState } from '../graph.ts';
import { z } from 'zod/v3';

export function createIdentifyIntentNode(llmClient: OpenRouterService) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    console.log(`🔍 Identifying intent...`, llmClient);
    const input = state.messages.at(-1)!.text;

    try {
      const systemPrompt = getSystemPrompt(professionals);
      const userPrompt = getUserPromptTemplate(input);
      const result = await llmClient.generateStructured(
        systemPrompt,
        userPrompt,
        IntentSchema as unknown as z.ZodSchema<any>
      )

      if (!result.success) {
        console.error('❌ Error in identifyIntent node:', result.error);
        return {
          ...state,
          intent: 'unknown',
          error: result.error,
        };
      }


      const intentData = result.data!
      console.log(`🔍 Intent Identified: ${intentData}`);

      return {
        ...intentData
      };
    } catch (error) {
      console.error('❌ Error in identifyIntent node:', error);
      return {
        ...state,
        intent: 'unknown',
        error: error instanceof Error ? error.message : 'Intent identification failed',
      };
    }
  };
}
