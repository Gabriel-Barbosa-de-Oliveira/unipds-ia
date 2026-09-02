import type { AIMessage, BaseMessage } from "@langchain/core/messages";

import type { ToolName, TraceEvent } from "./types.ts";

function contentToText(content: BaseMessage["content"]): string {
  if (typeof content === "string") {
    return content;
  }
  return content.map((part) => ("text" in part ? part.text : JSON.stringify(part))).join("");
}

function parseToolResult(content: BaseMessage["content"]): unknown {
  const text = contentToText(content);
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/** Mapeia mensagens de um agente LangChain/LangGraph (AI/tool) para a união TraceEvent comum. */
export function messagesToTrace(messages: BaseMessage[], startAt = 0): TraceEvent[] {
  const trace: TraceEvent[] = [];
  let at = startAt;

  for (const message of messages) {
    const role = message.getType();

    if (role === "ai") {
      const ai = message as AIMessage;
      const text = contentToText(ai.content).trim();
      const toolCalls = ai.tool_calls ?? [];

      if (toolCalls.length > 0) {
        if (text.length > 0) {
          trace.push({ type: "thought", at: at++, content: text });
        }
        for (const call of toolCalls) {
          trace.push({ type: "action", at: at++, tool: call.name as ToolName, args: call.args });
        }
      } else if (text.length > 0) {
        trace.push({ type: "answer", at: at++, content: text });
      }
    } else if (role === "tool") {
      trace.push({ type: "observation", at: at++, result: parseToolResult(message.content) });
    }
  }

  return trace;
}

/** Encontra o conteúdo do último evento `answer` em um trace, se houver. */
export function lastAnswer(trace: readonly TraceEvent[]): string | undefined {
  for (let i = trace.length - 1; i >= 0; i -= 1) {
    const event = trace[i];
    if (event?.type === "answer") {
      return event.content;
    }
  }
  return undefined;
}
