export type ToolName = "list_alerts" | "open_incident" | "resolve_incident";

export type TraceEvent =
  | { type: "thought"; at: number; content: string }
  | {
      type: "action";
      at: number;
      tool: ToolName;
      args: Record<string, unknown>;
    }
  | { type: "observation"; at: number; result: unknown }
  | { type: "plan"; at: number; steps: string[] }
  | { type: "critique"; at: number; content: string }
  | { type: "answer"; at: number; content: string };

export interface Metrics {
  llmCalls: number;
  latencyMs: number;
}

export interface RunResult {
  answer: string;
  trace: TraceEvent[];
  metrics: Metrics;
}

export interface RunOptions {
  maxIterations?: number;
}

export interface ReasoningStrategy {
  readonly name: string;
  run(input: string, options?: RunOptions): Promise<RunResult>;
}
