import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DashboardComponent } from './dashboard.component';

describe('DashboardComponent', () => {
  let fixture: ComponentFixture<DashboardComponent>;
  let compiled: HTMLElement;

  const button = () => compiled.querySelector('button') as HTMLButtonElement;

  const submit = async (value: string) => {
    const textarea = compiled.querySelector('textarea') as HTMLTextAreaElement;
    textarea.value = value;
    textarea.dispatchEvent(new Event('input'));
    compiled.querySelector('form')?.dispatchEvent(new Event('submit'));
    await fixture.whenStable();
  };

  beforeEach(async () => {
    vi.useFakeTimers({ toFake: ['setTimeout'] });
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(DashboardComponent);
    compiled = fixture.nativeElement;
    await fixture.whenStable();
  });

  afterEach(() => {
    vi.useRealTimers();
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

    vi.advanceTimersByTime(1500);
    await fixture.whenStable();

    expect(button().disabled).toBe(false);
    expect(button().textContent).toContain('Destilar Conquista');

    const card = compiled.querySelector('section a');
    expect(card?.getAttribute('href')).toBe('/detail/1');
    expect(card?.textContent).toContain('Melhorei o deploy');
  });
});
