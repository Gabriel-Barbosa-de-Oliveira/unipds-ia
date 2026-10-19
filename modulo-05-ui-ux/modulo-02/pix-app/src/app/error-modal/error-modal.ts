import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  afterNextRender,
  input,
  output,
  viewChild,
} from '@angular/core';

let nextId = 0;

/**
 * Modal de erro acessível baseado no <dialog> nativo.
 *
 * Uso: renderize condicionalmente e remova ao receber `closed`.
 *   @if (erro(); as e) {
 *     <app-error-modal [title]="e.title" [message]="e.message" (closed)="erro.set(null)" />
 *   }
 */
@Component({
  selector: 'app-error-modal',
  templateUrl: './error-modal.html',
  styleUrl: './error-modal.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    // `title` é um atributo global do HTML: sem isto o host exibiria tooltip
    // e leitores de tela poderiam anunciar o título em duplicidade.
    '[attr.title]': 'null',
  },
})
export class ErrorModal implements OnDestroy {
  readonly title = input.required<string>();
  readonly message = input.required<string>();

  readonly closed = output<void>();

  protected readonly titleId = `error-modal-title-${nextId}`;
  protected readonly messageId = `error-modal-message-${nextId++}`;

  private readonly dialog = viewChild.required<ElementRef<HTMLDialogElement>>('dialog');
  private previouslyFocused: HTMLElement | null = null;

  constructor() {
    afterNextRender(() => {
      this.previouslyFocused = document.activeElement as HTMLElement | null;
      // showModal() torna o resto da página inerte, prende o foco no diálogo
      // e dispara `cancel` ao pressionar ESC.
      this.dialog().nativeElement.showModal();
    });
  }

  /** Fechamento pelo botão. */
  protected close(): void {
    this.dialog().nativeElement.close();
  }

  /** ESC: o navegador dispara `cancel`; delegamos ao mesmo fluxo do botão. */
  protected onCancel(event: Event): void {
    event.preventDefault();
    this.close();
  }

  /** Disparado pelo <dialog> em qualquer forma de fechamento. */
  protected onClose(): void {
    this.restoreFocus();
    this.closed.emit();
  }

  ngOnDestroy(): void {
    this.restoreFocus();
  }

  private restoreFocus(): void {
    this.previouslyFocused?.focus();
    this.previouslyFocused = null;
  }
}
