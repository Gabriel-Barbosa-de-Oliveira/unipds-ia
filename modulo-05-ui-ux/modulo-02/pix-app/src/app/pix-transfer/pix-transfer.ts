import { ChangeDetectionStrategy, Component, signal } from '@angular/core';
import { FormField, form, min, required, submit } from '@angular/forms/signals';
import { ErrorModal } from '../error-modal/error-modal';

const LIMITE_PIX_SEM_APROVACAO = 5000;

interface PixTransfer {
  chavePix: string;
  valor: number;
  dataAgendamento: string;
}

@Component({
  selector: 'app-pix-transfer',
  imports: [FormField, ErrorModal],
  templateUrl: './pix-transfer.html',
  styleUrl: './pix-transfer.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PixTransferComponent {
  protected readonly model = signal<PixTransfer>({
    chavePix: '',
    valor: 0,
    dataAgendamento: '',
  });

  protected readonly pixForm = form(this.model, (p) => {
    required(p.chavePix, { message: 'Informe a chave Pix.' });
    required(p.valor, { message: 'Informe o valor.' });
    min(p.valor, 0.01, { message: 'O valor deve ser maior que zero.' });
    required(p.dataAgendamento, { message: 'Informe a data de agendamento.' });
  });

  protected readonly confirmado = signal<PixTransfer | null>(null);
  protected readonly limiteExcedido = signal(false);

  protected confirmar(event: Event): void {
    event.preventDefault();
    submit(this.pixForm, async (f) => {
      const pix = f().value();
      if (pix.valor > LIMITE_PIX_SEM_APROVACAO) {
        this.confirmado.set(null);
        this.limiteExcedido.set(true);
        return;
      }
      this.confirmado.set(pix);
    });
  }
}
