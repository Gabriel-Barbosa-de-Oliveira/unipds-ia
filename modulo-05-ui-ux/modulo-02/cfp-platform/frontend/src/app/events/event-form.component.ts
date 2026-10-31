import { HttpErrorResponse } from '@angular/common/http';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  signal,
} from '@angular/core';
import { CreateEvent } from 'shared-types';
import { EventService } from './event.service';

export type EventFormStatus = 'idle' | 'submitting' | 'success' | 'error';
export type EventField = 'name' | 'address' | 'capacity' | 'date';

export const NAME_MAX_LENGTH = 120;
export const ADDRESS_MAX_LENGTH = 200;
const POSITIVE_INTEGER_PATTERN = /^[1-9]\d*$/;

@Component({
  selector: 'app-event-form',
  templateUrl: './event-form.component.html',
  styleUrl: './event-form.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class EventFormComponent {
  private readonly eventService = inject(EventService);

  readonly name = signal('');
  readonly address = signal('');
  readonly capacity = signal('');
  readonly date = signal('');

  readonly status = signal<EventFormStatus>('idle');
  readonly serverMessage = signal('');
  readonly touched = signal<Record<EventField, boolean>>({
    name: false,
    address: false,
    capacity: false,
    date: false,
  });

  readonly errors = computed<Record<EventField, string | null>>(() => ({
    name: requiredText(this.name(), 'Nome', NAME_MAX_LENGTH),
    address: requiredText(this.address(), 'Endereço', ADDRESS_MAX_LENGTH),
    capacity: capacityError(this.capacity()),
    date: this.date() === '' ? 'Data é obrigatória.' : null,
  }));

  readonly isValid = computed(() =>
    Object.values(this.errors()).every((error) => error === null)
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

  showError(field: EventField): boolean {
    return this.touched()[field] && this.errors()[field] !== null;
  }

  onInput(field: EventField, event: Event): void {
    this[field].set((event.target as HTMLInputElement).value);
  }

  markTouched(field: EventField): void {
    this.touched.update((touched) => ({ ...touched, [field]: true }));
  }

  onSubmit(event: Event): void {
    event.preventDefault();
    this.touched.set({ name: true, address: true, capacity: true, date: true });
    if (!this.isValid() || this.status() === 'submitting') {
      return;
    }

    const newEvent: CreateEvent = {
      name: this.name(),
      address: this.address(),
      capacity: Number(this.capacity()),
      date: this.date(),
    };

    this.status.set('submitting');
    this.serverMessage.set('');
    this.eventService.create(newEvent).subscribe({
      next: (created) => {
        this.reset();
        this.serverMessage.set(
          `Evento "${created.name}" cadastrado com sucesso.`
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
    this.address.set('');
    this.capacity.set('');
    this.date.set('');
    this.touched.set({
      name: false,
      address: false,
      capacity: false,
      date: false,
    });
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

function capacityError(value: string): string | null {
  if (value === '') {
    return 'Capacidade é obrigatória.';
  }
  return POSITIVE_INTEGER_PATTERN.test(value)
    ? null
    : 'Capacidade deve ser um número inteiro maior que zero.';
}

function toErrorMessage(error: unknown): string {
  if (error instanceof HttpErrorResponse && error.status === 400) {
    const message = (error.error as { message?: string | string[] } | null)
      ?.message;
    const details = Array.isArray(message) ? message.join('; ') : message;
    return details
      ? `O servidor recusou o evento: ${details}`
      : 'O servidor recusou o evento.';
  }
  return 'Não foi possível cadastrar o evento. Tente novamente.';
}
