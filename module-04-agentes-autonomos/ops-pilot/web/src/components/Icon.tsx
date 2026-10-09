/** Ícones de traço (24×24, `currentColor`). Sempre decorativos: o texto ao lado carrega o sentido. */
const PATHS = {
  gear: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z",
  plus: "M12 5v14 M5 12h14",
  close: "M18 6 6 18 M6 6l12 12",
  copy: "M9 9h11v11H9z M5 15H4V4h11v1",
  warning: "M12 3 2 21h20L12 3Z M12 10v4 M12 17.5v.5",
  check: "M5 12.5l4.5 4.5L19 7",
  cross: "M7 7l10 10 M17 7 7 17",
  alert: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z M12 8v5 M12 16.5v.5",
  clock: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z M12 7v5l3 2",
  route: "M6 3v6a6 6 0 0 0 6 6h6 M15 12l3 3-3 3 M6 21v-6",
  thought: "M4 5h16v11H9l-5 4V5Z",
  plan: "M9 6h11 M9 12h11 M9 18h11 M4 6h.5 M4 12h.5 M4 18h.5",
  action: "M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.4-.6-.6-2.4 2.6-2.6Z",
  observation: "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6Z",
  critique: "M12 3l2.6 5.6 6.1.7-4.5 4.2 1.2 6L12 16.5 6.6 19.5l1.2-6L3.3 9.3l6.1-.7L12 3Z",
  fallback: "M4 7h13l-3-3 M20 17H7l3 3",
  answer: "M4 12.5l4.5 4.5L20 6",
  handoff: "M7 4 3 8l4 4 M3 8h12 M17 12l4 4-4 4 M21 16H9",
  unknown: "M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3 M12 17h.01 M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z",
  reasoning: "M12 3v3 M5.6 5.6l2.1 2.1 M3 12h3 M18 12h3 M16.3 7.7l2.1-2.1 M9 18h6 M10 21h4 M12 8a4 4 0 0 0-2.5 7.1V18h5v-2.9A4 4 0 0 0 12 8Z",
  spinner: "M12 3a9 9 0 1 0 9 9",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, className }: { name: IconName; className?: string }) {
  return (
    <svg
      className={className ? `icon ${className}` : "icon"}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
