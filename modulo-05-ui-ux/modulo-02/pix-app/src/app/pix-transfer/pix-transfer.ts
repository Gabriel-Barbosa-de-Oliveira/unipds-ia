import {
  ChangeDetectionStrategy,
  Component,
  Injector,
  afterNextRender,
  inject,
  signal,
} from '@angular/core';
import { FormField, form, min, required, submit } from '@angular/forms/signals';
import { ErrorModal } from '../error-modal/error-modal';
import { PixReceiptComponent } from '../pix-receipt/pix-receipt';

const LIMITE_PIX_SEM_APROVACAO = 5000;

interface PixTransfer {
  chavePix: string;
  valor: number;
  dataAgendamento: string;
}

interface PixConfirmado extends PixTransfer {
  confirmadoEm: Date;
}

const TRANSFERENCIA_VAZIA: PixTransfer = {
  chavePix: '',
  valor: 0,
  dataAgendamento: '',
};

@Component({
  selector: 'app-pix-transfer',
  imports: [FormField, ErrorModal, PixReceiptComponent],
  templateUrl: './pix-transfer.html',
  styleUrl: './pix-transfer.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PixTransferComponent {
  private readonly injector = inject(Injector);

  protected readonly model = signal<PixTransfer>({ ...TRANSFERENCIA_VAZIA });

  protected readonly pixForm = form(this.model, (p) => {
    required(p.chavePix, { message: 'Informe a chave Pix.' });
    required(p.valor, { message: 'Informe o valor.' });
    min(p.valor, 0.01, { message: 'O valor deve ser maior que zero.' });
    required(p.dataAgendamento, { message: 'Informe a data de agendamento.' });
  });

  protected readonly confirmado = signal<PixConfirmado | null>(null);
  protected readonly limiteExcedido = signal(false);

  protected confirmar(event: Event): void {
    event.preventDefault();
    submit(this.pixForm, async (f) => {
      const pix = f().value();
      if (pix.valor > LIMITE_PIX_SEM_APROVACAO) {
        this.limiteExcedido.set(true);
        return;
      }
      this.confirmado.set({ ...pix, confirmadoEm: new Date() });
    });
  }

  /** Volta do comprovante para um formulário limpo, devolvendo o foco ao primeiro campo. */
  protected novaTransferencia(): void {
    this.confirmado.set(null);
    this.pixForm().reset({ ...TRANSFERENCIA_VAZIA });
    afterNextRender(() => this.pixForm.chavePix().focusBoundControl(), {
      injector: this.injector,
    });
  }
}
