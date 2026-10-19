import { ChangeDetectionStrategy, Component, computed, inject, output, signal } from '@angular/core';
import { Location } from '@angular/common';

export type TransactionType = 'received' | 'sent';

export interface Transaction {
  id: string;
  title: string;
  /** Valor absoluto; o sinal é definido por `type`. */
  amount: number;
  type: TransactionType;
  date: Date;
}

const formatoMoeda = new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL' });
const formatoDia = new Intl.DateTimeFormat('pt-BR', { day: 'numeric', month: 'long' });
const formatoHora = new Intl.DateTimeFormat('pt-BR', { hour: '2-digit', minute: '2-digit' });

const TIPOS = {
  received: { sinal: '+', icone: 'arrow_upward', rotulo: 'Recebido' },
  sent: { sinal: '-', icone: 'arrow_downward', rotulo: 'Enviado' },
} as const;

@Component({
  selector: 'app-pix-history',
  templateUrl: './pix-history.html',
  styleUrl: './pix-history.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PixHistoryComponent {
  private readonly location = inject(Location);

  readonly gerarRelatorio = output<void>();
  readonly maisOpcoes = output<void>();

  protected readonly transactions = signal<Transaction[]>([
    {
      id: 'e2e-001',
      title: 'Pix recebido - Erick S.',
      amount: 500,
      type: 'received',
      date: new Date(2026, 2, 14, 14, 30),
    },
    {
      id: 'e2e-002',
      title: 'Transferência enviada - Pagamentos S/A',
      amount: 150,
      type: 'sent',
      date: new Date(2026, 2, 13, 9, 15),
    },
    {
      id: 'e2e-003',
      title: 'Pix recebido - Loja Central',
      amount: 1250,
      type: 'received',
      date: new Date(2026, 2, 12, 18, 45),
    },
    {
      id: 'e2e-004',
      title: 'Pagamento efetuado - QR Code',
      amount: 42.9,
      type: 'sent',
      date: new Date(2026, 2, 12, 10, 20),
    },
  ]);

  /** Dados já formatados para manter o template livre de lógica. */
  protected readonly items = computed(() =>
    this.transactions().map((t) => ({
      ...t,
      ...TIPOS[t.type],
      valorFormatado: `${TIPOS[t.type].sinal} ${formatoMoeda.format(t.amount)}`,
      dataFormatada: `${formatoDia.format(t.date)}, ${formatoHora.format(t.date)}`,
      dataIso: t.date.toISOString(),
    })),
  );

  protected voltar(): void {
    this.location.back();
  }
}
