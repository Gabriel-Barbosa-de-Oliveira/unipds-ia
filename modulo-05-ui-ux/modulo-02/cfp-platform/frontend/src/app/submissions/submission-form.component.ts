import { HttpErrorResponse } from '@angular/common/http';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  signal,
} from '@angular/core';
import { CreateSpeakerSubmission } from 'shared-types';
import { SubmissionService } from './submission.service';

export type SubmissionStatus = 'idle' | 'submitting' | 'success' | 'error';
export type TextField = 'name' | 'email' | 'talkTitle';

export const NAME_MAX_LENGTH = 120;
export const TALK_TITLE_MAX_LENGTH = 200;
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

@Component({
  selector: 'app-submission-form',
  templateUrl: './submission-form.component.html',
  styleUrl: './submission-form.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SubmissionFormComponent {
  private readonly submissionService = inject(SubmissionService);

  readonly name = signal('');
  readonly email = signal('');
  readonly talkTitle = signal('');
  readonly isGDE = signal(false);

  readonly status = signal<SubmissionStatus>('idle');
  readonly serverMessage = signal('');
  readonly touched = signal<Record<TextField, boolean>>({
    name: false,
    email: false,
    talkTitle: false,
  });

  readonly errors = computed<Record<TextField, string | null>>(() => ({
    name: requiredText(this.name(), 'Nome', NAME_MAX_LENGTH),
    email: EMAIL_PATTERN.test(this.email())
      ? null
      : 'Informe um e-mail válido.',
    talkTitle: requiredText(
      this.talkTitle(),
      'Título da palestra',
      TALK_TITLE_MAX_LENGTH
    ),
  }));

  readonly isValid = computed(() =>
    Object.values(this.errors()).every((error) => error === null)
  );

  readonly canSubmit = computed(
    () => this.isValid() && this.status() !== 'submitting'
  );

  readonly statusMessage = computed(() => {
    switch (this.status()) {
      case 'submitting':
        return 'Enviando…';
      case 'success':
      case 'error':
        return this.serverMessage();
      default:
        return '';
    }
  });

  protected readonly nameMaxLength = NAME_MAX_LENGTH;
  protected readonly talkTitleMaxLength = TALK_TITLE_MAX_LENGTH;

  showError(field: TextField): boolean {
    return this.touched()[field] && this.errors()[field] !== null;
  }

  onTextInput(field: TextField, event: Event): void {
    const value = (event.target as HTMLInputElement).value;
    this[field].set(value);
  }

  onGdeChange(event: Event): void {
    this.isGDE.set((event.target as HTMLInputElement).checked);
  }

  markTouched(field: TextField): void {
    this.touched.update((touched) => ({ ...touched, [field]: true }));
  }

  onSubmit(event: Event): void {
    event.preventDefault();
    this.touched.set({ name: true, email: true, talkTitle: true });
    if (!this.canSubmit()) {
      return;
    }

    const submission: CreateSpeakerSubmission = {
      name: this.name(),
      email: this.email(),
      talkTitle: this.talkTitle(),
      isGDE: this.isGDE(),
    };

    this.status.set('submitting');
    this.serverMessage.set('');
    this.submissionService.submit(submission).subscribe({
      next: (created) => {
        this.reset();
        this.serverMessage.set(
          `Proposta "${created.talkTitle}" enviada com sucesso.`
        );
        this.status.set('success');
      },
      error: (error: unknown) => {
        this.serverMessage.set(toErrorMessage(error));
        this.status.set('error');
      },
    });
  }

  private reset(): void {
    this.name.set('');
    this.email.set('');
    this.talkTitle.set('');
    this.isGDE.set(false);
    this.touched.set({ name: false, email: false, talkTitle: false });
  }
}

function requiredText(
  value: string,
  label: string,
  maxLength: number
): string | null {
  if (value.trim().length === 0) {
    return `${label} é obrigatório.`;
  }
  if (value.length > maxLength) {
    return `${label} deve ter no máximo ${maxLength} caracteres.`;
  }
  return null;
}

function toErrorMessage(error: unknown): string {
  if (error instanceof HttpErrorResponse && error.status === 400) {
    const message = (error.error as { message?: string | string[] } | null)
      ?.message;
    const details = Array.isArray(message) ? message.join('; ') : message;
    return details
      ? `O servidor recusou a proposta: ${details}`
      : 'O servidor recusou a proposta.';
  }
  return 'Não foi possível enviar sua proposta. Tente novamente.';
}
