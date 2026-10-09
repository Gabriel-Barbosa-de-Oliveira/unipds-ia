import { Icon } from "./Icon.tsx";

interface HeaderProps {
  onNewConversation: () => void;
  onOpenSettings: () => void;
  canReset: boolean;
}

export function Header({ onNewConversation, onOpenSettings, canReset }: HeaderProps) {
  return (
    <header className="header">
      <h1>
        OpsPilot <span className="header-sub">· War room</span>
      </h1>
      <button type="button" className="btn btn-ghost" onClick={onNewConversation} disabled={!canReset}>
        <Icon name="plus" />
        Nova conversa
      </button>
      <button type="button" className="btn btn-ghost btn-icon" onClick={onOpenSettings} aria-label="Configurações">
        <Icon name="gear" />
      </button>
    </header>
  );
}
