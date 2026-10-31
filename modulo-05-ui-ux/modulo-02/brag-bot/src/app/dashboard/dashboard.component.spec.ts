import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { DashboardComponent } from './dashboard.component';

describe('DashboardComponent', () => {
  let fixture: ComponentFixture<DashboardComponent>;
  let compiled: HTMLElement;
  let http: HttpTestingController;

  const button = () => compiled.querySelector('button') as HTMLButtonElement;

  const submit = async (value: string) => {
    const textarea = compiled.querySelector('textarea') as HTMLTextAreaElement;
    textarea.value = value;
    textarea.dispatchEvent(new Event('input'));
    compiled.querySelector('form')?.dispatchEvent(new Event('submit'));
    await fixture.whenStable();
  };

  const settle = async () => {
    await new Promise((resolve) => setTimeout(resolve));
    await fixture.whenStable();
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [provideRouter([]), provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(DashboardComponent);
    compiled = fixture.nativeElement;
    await fixture.whenStable();
  });

  afterEach(() => {
    http.verify();
  });

  it('should reject a blank achievement', async () => {
    await submit('   ');

    expect(compiled.textContent).toContain('Descreva sua conquista antes de destilar.');
    expect(button().disabled).toBe(false);
  });

  it('should show the loading state on the button and then render a card', async () => {
    expect(button().textContent).toContain('Destilar Conquista');
    expect(compiled.textContent).toContain('Nenhuma conquista destilada ainda');

    await submit('Melhorei o deploy');
    expect(button().disabled).toBe(true);
    expect(button().textContent).toContain('Destilando...');

    http.expectOne('/api/brag').flush({
      id: 'abc',
      title: 'Deploy mais rápido',
      context: 'Pipeline lento',
      actionTaken: 'Paralelizei os testes',
      businessImpact: 'Time mais ágil',
      metrics: ['19 minutos por deploy'],
      technologiesUsed: ['Docker'],
    });
    await settle();

    expect(button().disabled).toBe(false);
    expect(button().textContent).toContain('Destilar Conquista');

    const card = compiled.querySelector('section a');
    expect(card?.getAttribute('href')).toBe('/detail/abc');
    expect(card?.textContent).toContain('Melhorei o deploy');
    expect(card?.textContent).toContain('Docker');
    expect((compiled.querySelector('textarea') as HTMLTextAreaElement).value).toBe('');
  });

  it('should show the error and keep the draft when generation fails', async () => {
    await submit('Melhorei o deploy');

    http.expectOne('/api/brag').flush(null, { status: 500, statusText: 'Server Error' });
    await settle();

    expect(button().disabled).toBe(false);
    expect(compiled.querySelector('[role="alert"]')?.textContent).toContain(
      'Não foi possível destilar',
    );
    expect((compiled.querySelector('textarea') as HTMLTextAreaElement).value).toBe(
      'Melhorei o deploy',
    );
  });
});
