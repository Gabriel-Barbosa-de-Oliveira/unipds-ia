import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  afterNextRender,
  computed,
  inject,
  input,
  output,
  signal,
  viewChild,
} from '@angular/core';

const formatoMoeda = new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL' });
const formatoDataHora = new Intl.DateTimeFormat('pt-BR', { dateStyle: 'long', timeStyle: 'medium' });
const TEMPO_FEEDBACK_COPIA_MS = 2200;

@Component({
  selector: 'app-pix-receipt',
  templateUrl: './pix-receipt.html',
  styleUrl: './pix-receipt.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PixReceiptComponent {
  readonly valor = input.required<number>();
  readonly nome = input.required<string>();
  readonly dataHora = input<Date>(new Date());
  readonly tipoConta = input<string>();
  readonly cpf = input<string>();
  readonly instituicao = input<string>();
  readonly tipoChave = input<string>();
  readonly idTransacao = input<string>();

  readonly voltar = output<void>();
  readonly compartilhar = output<void>();
  readonly baixarPdf = output<void>();

  protected readonly valorFormatado = computed(() => formatoMoeda.format(this.valor()));
  protected readonly dataHoraFormatada = computed(() => formatoDataHora.format(this.dataHora()));
  protected readonly dataHoraIso = computed(() => this.dataHora().toISOString());

  protected readonly copiado = signal(false);

  private readonly titulo = viewChild.required<ElementRef<HTMLHeadingElement>>('titulo');
  private timerCopia?: ReturnType<typeof setTimeout>;

  constructor() {
    // O comprovante substitui o formulário: leva o foco ao título para que
    // leitores de tela anunciem a nova tela em vez de perder o contexto.
    afterNextRender(() => this.titulo().nativeElement.focus());
    inject(DestroyRef).onDestroy(() => clearTimeout(this.timerCopia));
  }

  protected async copiarId(): Promise<void> {
    const id = this.idTransacao();
    if (!id) return;

    await navigator.clipboard.writeText(id);
    this.copiado.set(true);
    clearTimeout(this.timerCopia);
    this.timerCopia = setTimeout(() => this.copiado.set(false), TEMPO_FEEDBACK_COPIA_MS);
  }
}
