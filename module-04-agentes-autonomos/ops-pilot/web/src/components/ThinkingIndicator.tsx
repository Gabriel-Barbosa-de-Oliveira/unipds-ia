import { useEffect, useState } from "react";

/** Abaixo disso o indicador não aparece, para não piscar (design.instructions.md). */
const SHOW_AFTER_MS = 300;

/** Skeleton com o tempo decorrido enquanto o copiloto responde (FR-003). */
export function ThinkingIndicator() {
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    const startedAt = Date.now();
    const tick = () => setElapsedMs(Date.now() - startedAt);
    const first = window.setTimeout(tick, SHOW_AFTER_MS);
    const interval = window.setInterval(tick, 1000);
    return () => {
      window.clearTimeout(first);
      window.clearInterval(interval);
    };
  }, []);

  if (elapsedMs < SHOW_AFTER_MS) {
    return null;
  }

  return (
    <div className="thinking" aria-busy="true">
      <span className="thinking-label">Pensando… {Math.floor(elapsedMs / 1000)}s</span>
      <div className="skeleton-line" />
      <div className="skeleton-line" />
      <div className="skeleton-line" />
    </div>
  );
}
