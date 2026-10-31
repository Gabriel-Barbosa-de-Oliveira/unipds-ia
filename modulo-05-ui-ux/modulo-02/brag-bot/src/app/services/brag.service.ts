import { HttpClient } from '@angular/common/http';
import { inject, Injectable, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';

export interface Brag {
  id: string;
  prompt: string;
  title: string;
  context: string;
  actionTaken: string;
  businessImpact: string;
  metrics: string[];
  technologiesUsed: string[];
}

@Injectable({ providedIn: 'root' })
export class BragService {
  private readonly http = inject(HttpClient);

  readonly brags = signal<Brag[]>([]);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);

  async generateBrag(definition: string): Promise<boolean> {
    this.loading.set(true);
    this.error.set(null);

    try {
      const brag = await firstValueFrom(
        this.http.post<Omit<Brag, 'prompt'>>('/api/brag', { definition }),
      );
      this.brags.update((brags) => [{ ...brag, prompt: definition }, ...brags]);
      return true;
    } catch {
      this.error.set('Não foi possível destilar sua conquista. Tente novamente.');
      return false;
    } finally {
      this.loading.set(false);
    }
  }
}
