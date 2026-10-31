import { Injectable, signal } from '@angular/core';

export interface Brag {
  id: string;
  prompt: string;
  title: string;
  context: string;
  impact: string;
  metrics: string[];
  technologies: string[];
}

const MOCK_LATENCY_MS = 1500;

const MOCK_BRAG: Omit<Brag, 'id' | 'prompt'> = {
  title: 'Redução de 40% no tempo de deploy da plataforma',
  context:
    'O pipeline de entrega levava mais de 30 minutos e travava o time em dias de release. ' +
    'Liderei a revisão das etapas de build e testes, paralelizando o que era sequencial.',
  impact:
    'Os deploys ficaram mais rápidos e previsíveis, liberando o time para entregar ' +
    'pequenas mudanças várias vezes ao dia, com menos risco a cada release.',
  metrics: [
    'Tempo de deploy caiu de 32 para 19 minutos',
    'Frequência de releases subiu de 2 para 8 por semana',
    'Falhas de pipeline reduzidas em 25%',
  ],
  technologies: ['GitHub Actions', 'Docker', 'Node.js', 'Terraform'],
};

@Injectable({ providedIn: 'root' })
export class BragService {
  readonly brags = signal<Brag[]>([]);
  readonly loading = signal(false);

  generateMockBrag(prompt: string): void {
    this.loading.set(true);

    setTimeout(() => {
      this.brags.update((brags) => [
        { ...MOCK_BRAG, id: String(brags.length + 1), prompt },
        ...brags,
      ]);
      this.loading.set(false);
    }, MOCK_LATENCY_MS);
  }
}
