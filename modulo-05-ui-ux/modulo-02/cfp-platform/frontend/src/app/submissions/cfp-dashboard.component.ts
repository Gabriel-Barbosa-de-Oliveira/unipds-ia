import { HttpClient } from '@angular/common/http';
import {
  ChangeDetectionStrategy,
  Component,
  WritableSignal,
  computed,
  inject,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { RouterLink } from '@angular/router';
import { SpeakerDTO } from 'shared-types';

export type DashboardStatus = 'loading' | 'loaded' | 'error';

@Component({
  selector: 'app-cfp-dashboard',
  imports: [RouterLink],
  templateUrl: './cfp-dashboard.component.html',
  styleUrl: './cfp-dashboard.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CfpDashboardComponent {
  private readonly http = inject(HttpClient);

  readonly speakers: WritableSignal<SpeakerDTO[]> = signal([]);
  readonly status = signal<DashboardStatus>('loading');

  readonly isEmpty = computed(
    () => this.status() === 'loaded' && this.speakers().length === 0
  );

  readonly statusMessage = computed(() => {
    switch (this.status()) {
      case 'loading':
        return 'Carregando propostas…';
      case 'error':
        return 'Não foi possível carregar as propostas. Tente novamente.';
      default: {
        const count = this.speakers().length;
        return count === 1
          ? '1 proposta enviada.'
          : `${count} propostas enviadas.`;
      }
    }
  });

  constructor() {
    this.http
      .get<SpeakerDTO[]>('/api/submissions')
      .pipe(takeUntilDestroyed())
      .subscribe({
        next: (speakers) => {
          this.speakers.set(speakers);
          this.status.set('loaded');
        },
        error: () => this.status.set('error'),
      });
  }
}
