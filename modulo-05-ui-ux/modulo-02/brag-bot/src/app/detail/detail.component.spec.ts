import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { BragService } from '../services/brag.service';
import { DetailComponent } from './detail.component';

describe('DetailComponent', () => {
  let fixture: ComponentFixture<DetailComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DetailComponent],
      providers: [provideRouter([])],
    }).compileComponents();

    TestBed.inject(BragService).brags.set([
      {
        id: '1',
        prompt: 'Melhorei o deploy',
        title: 'Deploy mais rápido',
        context: 'Pipeline lento',
        impact: 'Time mais ágil',
        metrics: ['19 minutos por deploy'],
        technologies: ['Docker'],
      },
    ]);

    fixture = TestBed.createComponent(DetailComponent);
  });

  it('should render every section of the brag', async () => {
    fixture.componentRef.setInput('id', '1');
    await fixture.whenStable();

    const text = (fixture.nativeElement as HTMLElement).textContent;
    for (const expected of [
      'Deploy mais rápido',
      'Pipeline lento',
      'Time mais ágil',
      '19 minutos por deploy',
      'Docker',
      'Voltar ao Dashboard',
    ]) {
      expect(text).toContain(expected);
    }
  });

  it('should show a not found state for an unknown id', async () => {
    fixture.componentRef.setInput('id', '99');
    await fixture.whenStable();

    const text = (fixture.nativeElement as HTMLElement).textContent;
    expect(text).toContain('Conquista não encontrada');
    expect(text).toContain('Voltar ao Dashboard');
  });
});
