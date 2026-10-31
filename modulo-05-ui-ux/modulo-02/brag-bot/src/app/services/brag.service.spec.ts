import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BragService } from './brag.service';

describe('BragService', () => {
  let service: BragService;

  beforeEach(() => {
    vi.useFakeTimers();
    service = TestBed.inject(BragService);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should stay loading until the simulated request finishes', () => {
    service.generateMockBrag('Melhorei o deploy');

    expect(service.loading()).toBe(true);
    expect(service.brags()).toEqual([]);

    vi.advanceTimersByTime(1499);
    expect(service.loading()).toBe(true);

    vi.advanceTimersByTime(1);
    expect(service.loading()).toBe(false);
    expect(service.brags()).toHaveLength(1);
  });

  it('should add the mocked brag with all of its fields', () => {
    service.generateMockBrag('Melhorei o deploy');
    vi.advanceTimersByTime(1500);

    const [brag] = service.brags();
    expect(brag.id).toBe('1');
    expect(brag.prompt).toBe('Melhorei o deploy');
    expect(brag.title).toBeTruthy();
    expect(brag.context).toBeTruthy();
    expect(brag.impact).toBeTruthy();
    expect(brag.metrics.length).toBeGreaterThan(0);
    expect(brag.technologies.length).toBeGreaterThan(0);
  });

  it('should give each brag its own id and keep the newest first', () => {
    service.generateMockBrag('Primeira');
    vi.advanceTimersByTime(1500);
    service.generateMockBrag('Segunda');
    vi.advanceTimersByTime(1500);

    expect(service.brags().map(({ id, prompt }) => ({ id, prompt }))).toEqual([
      { id: '2', prompt: 'Segunda' },
      { id: '1', prompt: 'Primeira' },
    ]);
  });
});
