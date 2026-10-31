import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';
import { Title } from '@angular/platform-browser';
import { CreateEvent, EventDTO } from 'shared-types';
import { appRoutes } from '../app.routes';
import { EventFormComponent } from './event-form.component';

const validEvent: CreateEvent = {
  name: 'Angular Summit',
  address: 'Av. Paulista, 1000',
  capacity: 200,
  date: '2027-03-15',
};

describe('EventFormComponent', () => {
  let fixture: ComponentFixture<EventFormComponent>;
  let component: EventFormComponent;
  let element: HTMLElement;
  let httpTesting: HttpTestingController;

  function query<T extends Element>(selector: string): T {
    const found = element.querySelector<T>(selector);
    if (!found) {
      throw new Error(`Element not found: ${selector}`);
    }
    return found;
  }

  const submitButton = () => query<HTMLButtonElement>('button[type="submit"]');
  const input = (name: string) => query<HTMLInputElement>(`input[name="${name}"]`);
  const liveRegion = () => query<HTMLElement>('[role="status"]');
  const form = () => query<HTMLFormElement>('form');
  const errorText = (name: string) =>
    element.querySelector(`#event-${name}-error`)?.textContent?.trim();

  function typeInto(name: string, value: string): void {
    const field = input(name);
    field.value = value;
    field.dispatchEvent(new Event('input'));
    field.dispatchEvent(new Event('blur'));
    fixture.detectChanges();
  }

  function fillValidForm(): void {
    typeInto('name', validEvent.name);
    typeInto('address', validEvent.address);
    typeInto('capacity', String(validEvent.capacity));
    typeInto('date', validEvent.date);
  }

  function submit(): void {
    form().dispatchEvent(new Event('submit', { cancelable: true }));
    fixture.detectChanges();
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EventFormComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(EventFormComponent);
    component = fixture.componentInstance;
    element = fixture.nativeElement;
    httpTesting = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
  });

  afterEach(() => httpTesting.verify());

  describe('initial state', () => {
    it('should render a heading and the four empty fields with labels', () => {
      expect(query('h1').textContent).toContain('Cadastre um evento');
      for (const name of ['name', 'address', 'capacity', 'date']) {
        const field = input(name);
        expect(field.id).toBe(`event-${name}`);
        expect(field.value).toBe('');
        expect(query(`label[for="event-${name}"]`)).toBeTruthy();
      }
    });

    it('should use numeric and date input types', () => {
      expect(input('capacity').type).toBe('number');
      expect(input('date').type).toBe('date');
    });

    it('should enable the submit button while empty', () => {
      expect(submitButton().disabled).toBe(false);
    });

    it('should mark every field as natively required', () => {
      for (const name of ['name', 'address', 'capacity', 'date']) {
        expect(input(name).required).toBe(true);
      }
    });

    it('should not show errors before any interaction', () => {
      expect(element.querySelectorAll('.field-error').length).toBe(0);
    });
  });

  describe('validation', () => {
    it('should show all required messages and skip the API when submitted empty', () => {
      submit();

      expect(errorText('name')).toBe('Nome é obrigatório.');
      expect(errorText('address')).toBe('Endereço é obrigatório.');
      expect(errorText('capacity')).toBe('Capacidade é obrigatória.');
      expect(errorText('date')).toBe('Data é obrigatória.');
      httpTesting.expectNone('/api/events');
    });

    it('should link the error to its field for assistive technology', () => {
      submit();

      expect(input('name').getAttribute('aria-invalid')).toBe('true');
      expect(input('name').getAttribute('aria-describedby')).toBe(
        'event-name-error'
      );
    });

    it('should treat whitespace-only text as empty', () => {
      fillValidForm();
      typeInto('address', '   ');
      submit();

      expect(errorText('address')).toBe('Endereço é obrigatório.');
      httpTesting.expectNone('/api/events');
    });

    it.each(['0', '-1', '1.5'])('should reject capacity %s', (capacity) => {
      fillValidForm();
      typeInto('capacity', capacity);
      submit();

      expect(errorText('capacity')).toBe(
        'Capacidade deve ser um número inteiro maior que zero.'
      );
      httpTesting.expectNone('/api/events');
    });

    it('should reject text longer than the API limits', () => {
      fillValidForm();
      typeInto('name', 'a'.repeat(121));
      typeInto('address', 'a'.repeat(201));
      submit();

      expect(errorText('name')).toBe('Nome deve ter no máximo 120 caracteres.');
      expect(errorText('address')).toBe(
        'Endereço deve ter no máximo 200 caracteres.'
      );
      httpTesting.expectNone('/api/events');
    });

    it('should remove the error once the field is corrected', () => {
      submit();
      typeInto('name', validEvent.name);

      expect(errorText('name')).toBeUndefined();
      expect(input('name').getAttribute('aria-invalid')).toBeNull();
    });
  });

  describe('submission', () => {
    it('should POST the event with a numeric capacity', () => {
      fillValidForm();
      submit();

      const req = httpTesting.expectOne('/api/events');
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(validEvent);
      req.flush({ id: 'abc-123', ...validEvent } satisfies EventDTO);
    });

    it('should disable the button and announce progress while submitting', () => {
      fillValidForm();
      submit();

      expect(submitButton().disabled).toBe(true);
      expect(liveRegion().textContent).toContain('Enviando…');
      httpTesting.expectOne('/api/events').flush({ id: '1', ...validEvent });
    });

    it('should announce success and clear the form', () => {
      fillValidForm();
      submit();
      httpTesting.expectOne('/api/events').flush({ id: '1', ...validEvent });
      fixture.detectChanges();

      expect(liveRegion().textContent).toContain(
        'Evento "Angular Summit" cadastrado com sucesso.'
      );
      expect(liveRegion().classList).toContain('form-status--success');
      for (const name of ['name', 'address', 'capacity', 'date']) {
        expect(input(name).value).toBe('');
      }
      expect(element.querySelectorAll('.field-error').length).toBe(0);
      expect(submitButton().disabled).toBe(false);
    });

    it('should show the server message on 400 and keep the data', () => {
      fillValidForm();
      submit();
      httpTesting
        .expectOne('/api/events')
        .flush(
          { message: ['date must be a valid ISO 8601 date string'] },
          { status: 400, statusText: 'Bad Request' }
        );
      fixture.detectChanges();

      expect(liveRegion().textContent).toContain(
        'O servidor recusou o evento: date must be a valid ISO 8601 date string'
      );
      expect(liveRegion().classList).toContain('form-status--error');
      expect(input('name').value).toBe(validEvent.name);
      expect(component.status()).toBe('error');
    });

    it('should show a generic message on other failures and keep the data', () => {
      fillValidForm();
      submit();
      httpTesting
        .expectOne('/api/events')
        .flush(null, { status: 500, statusText: 'Server Error' });
      fixture.detectChanges();

      expect(liveRegion().textContent).toContain(
        'Não foi possível cadastrar o evento. Tente novamente.'
      );
      expect(input('address').value).toBe(validEvent.address);
      expect(submitButton().disabled).toBe(false);
    });
  });
});

describe('event/new route', () => {
  it('should render the form with its own document title', async () => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter(appRoutes),
      ],
    });

    const harness = await RouterTestingHarness.create();
    await harness.navigateByUrl('/event/new', EventFormComponent);

    expect(TestBed.inject(Router).url).toBe('/event/new');
    expect(TestBed.inject(Title).getTitle()).toBe('Cadastrar evento');
  });
});
