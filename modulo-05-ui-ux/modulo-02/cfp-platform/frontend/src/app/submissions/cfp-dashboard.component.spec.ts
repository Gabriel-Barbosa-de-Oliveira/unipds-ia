import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { SpeakerDTO } from 'shared-types';
import { CfpDashboardComponent } from './cfp-dashboard.component';

const speakers: SpeakerDTO[] = [
  {
    id: 'abc-1',
    name: 'Ada Lovelace',
    email: 'ada@example.com',
    talkTitle: 'Signals in practice',
    isGDE: true,
  },
  {
    id: 'abc-2',
    name: 'Grace Hopper',
    email: 'grace@example.com',
    talkTitle: 'Zoneless change detection',
    isGDE: false,
  },
];

describe('CfpDashboardComponent', () => {
  let fixture: ComponentFixture<CfpDashboardComponent>;
  let component: CfpDashboardComponent;
  let element: HTMLElement;
  let httpTesting: HttpTestingController;

  function query<T extends Element>(selector: string): T {
    const found = element.querySelector<T>(selector);
    if (!found) {
      throw new Error(`Element not found: ${selector}`);
    }
    return found;
  }

  const liveRegion = () => query<HTMLElement>('[role="status"]');
  const section = () => query<HTMLElement>('section');
  const table = () => element.querySelector('table');

  function respond(body: SpeakerDTO[]): void {
    httpTesting.expectOne('/api/submissions').flush(body);
    fixture.detectChanges();
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CfpDashboardComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(CfpDashboardComponent);
    component = fixture.componentInstance;
    element = fixture.nativeElement;
    httpTesting = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
  });

  afterEach(() => httpTesting.verify());

  describe('while loading', () => {
    it('should start in loading state with an empty Signal', () => {
      expect(component.status()).toBe('loading');
      expect(component.speakers()).toEqual([]);
      expect(section().getAttribute('aria-busy')).toBe('true');
      expect(liveRegion().textContent).toContain('Carregando');
      expect(liveRegion().getAttribute('aria-live')).toBe('polite');
      expect(table()).toBeNull();

      const req = httpTesting.expectOne('/api/submissions');
      expect(req.request.method).toBe('GET');
      req.flush([]);
    });
  });

  describe('with submissions', () => {
    beforeEach(() => respond(speakers));

    it('should store the response in the speakers Signal', () => {
      expect(component.status()).toBe('loaded');
      expect(component.speakers()).toEqual(speakers);
      expect(section().getAttribute('aria-busy')).toBe('false');
    });

    it('should render one row per submission, in order', () => {
      const rows = Array.from(element.querySelectorAll('tbody tr'));
      const cells = rows.map((row) =>
        Array.from(row.querySelectorAll('td')).map((td) =>
          td.textContent?.trim()
        )
      );

      expect(cells).toEqual([
        ['Ada Lovelace', 'ada@example.com', 'Signals in practice', 'Sim'],
        ['Grace Hopper', 'grace@example.com', 'Zoneless change detection', 'Não'],
      ]);
    });

    it('should expose an accessible table structure', () => {
      expect(table()?.querySelector('caption')?.textContent?.trim()).toBeTruthy();
      const headers = Array.from(element.querySelectorAll('thead th'));
      expect(headers.map((th) => th.textContent?.trim())).toEqual([
        'Nome',
        'E-mail',
        'Título da palestra',
        'GDE',
      ]);
      headers.forEach((th) => expect(th.getAttribute('scope')).toBe('col'));
    });
  });

  it('should show the empty state when there are no submissions', () => {
    respond([]);

    expect(component.isEmpty()).toBe(true);
    expect(table()).toBeNull();
    expect(query('.dashboard-empty').textContent).toContain(
      'Ainda não há propostas enviadas'
    );
  });

  it('should announce an error and hide the table when the request fails', () => {
    httpTesting
      .expectOne('/api/submissions')
      .flush(null, { status: 500, statusText: 'Internal Server Error' });
    fixture.detectChanges();

    expect(component.status()).toBe('error');
    expect(table()).toBeNull();
    expect(liveRegion().textContent).toContain(
      'Não foi possível carregar as propostas'
    );
  });

  it('should link back to the submission form', () => {
    respond([]);

    expect(query<HTMLAnchorElement>('a.button-link').getAttribute('href')).toBe(
      '/'
    );
  });
});
