import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { BragService } from './brag.service';

const apiBrag = (id: string) => ({
  id,
  title: 'Deploy mais rápido',
  context: 'Pipeline lento',
  actionTaken: 'Paralelizei os testes',
  businessImpact: 'Time mais ágil',
  metrics: ['19 minutos por deploy'],
  technologiesUsed: ['Docker'],
});

describe('BragService', () => {
  let service: BragService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(BragService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  it('should stay loading until the request finishes', async () => {
    const result = service.generateBrag('Melhorei o deploy');

    expect(service.loading()).toBe(true);
    expect(service.brags()).toEqual([]);

    http.expectOne('/api/brag').flush(apiBrag('a'));

    expect(await result).toBe(true);
    expect(service.loading()).toBe(false);
    expect(service.brags()).toHaveLength(1);
  });

  it('should post the definition and keep the draft alongside the API result', async () => {
    const result = service.generateBrag('Melhorei o deploy');

    const request = http.expectOne('/api/brag');
    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({ definition: 'Melhorei o deploy' });
    request.flush(apiBrag('a'));
    await result;

    expect(service.brags()).toEqual([{ ...apiBrag('a'), prompt: 'Melhorei o deploy' }]);
  });

  it('should keep the newest brag first', async () => {
    const first = service.generateBrag('Primeira');
    http.expectOne('/api/brag').flush(apiBrag('a'));
    await first;

    const second = service.generateBrag('Segunda');
    http.expectOne('/api/brag').flush(apiBrag('b'));
    await second;

    expect(service.brags().map(({ id, prompt }) => ({ id, prompt }))).toEqual([
      { id: 'b', prompt: 'Segunda' },
      { id: 'a', prompt: 'Primeira' },
    ]);
  });

  it('should expose an error and stop loading when the request fails', async () => {
    const result = service.generateBrag('Melhorei o deploy');

    http
      .expectOne('/api/brag')
      .flush({ error: 'Failed to generate brag' }, { status: 500, statusText: 'Server Error' });

    expect(await result).toBe(false);
    expect(service.loading()).toBe(false);
    expect(service.error()).toBeTruthy();
    expect(service.brags()).toEqual([]);
  });

  it('should clear the previous error on a new attempt', async () => {
    const failed = service.generateBrag('Melhorei o deploy');
    http.expectOne('/api/brag').flush(null, { status: 500, statusText: 'Server Error' });
    await failed;

    const retried = service.generateBrag('Melhorei o deploy');
    expect(service.error()).toBeNull();

    http.expectOne('/api/brag').flush(apiBrag('a'));
    await retried;
  });
});
