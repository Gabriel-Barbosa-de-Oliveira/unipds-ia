import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { CreateSpeakerSubmission } from 'shared-types';
import { SubmissionFormComponent } from './submission-form.component';

const validSubmission: CreateSpeakerSubmission = {
  name: 'Ada Lovelace',
  email: 'ada@example.com',
  talkTitle: 'Signals in practice',
  isGDE: false,
};

describe('SubmissionFormComponent', () => {
  let fixture: ComponentFixture<SubmissionFormComponent>;
  let component: SubmissionFormComponent;
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
  const input = (id: string) => query<HTMLInputElement>(`#${id}`);
  const liveRegion = () => query<HTMLElement>('[role="status"]');
  const form = () => query<HTMLFormElement>('form');

  function typeInto(id: string, value: string): void {
    const field = input(id);
    field.value = value;
    field.dispatchEvent(new Event('input'));
    field.dispatchEvent(new Event('blur'));
    fixture.detectChanges();
  }

  function fillValidForm(): void {
    typeInto('submission-name', validSubmission.name);
    typeInto('submission-email', validSubmission.email);
    typeInto('submission-talk-title', validSubmission.talkTitle);
  }

  function submit(): void {
    form().dispatchEvent(new Event('submit', { cancelable: true }));
    fixture.detectChanges();
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SubmissionFormComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(SubmissionFormComponent);
    component = fixture.componentInstance;
    element = fixture.nativeElement;
    httpTesting = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
  });

  afterEach(() => httpTesting.verify());

  describe('initial Signal state', () => {
    it('should start with empty fields, isGDE false and idle status', () => {
      expect(component.name()).toBe('');
      expect(component.email()).toBe('');
      expect(component.talkTitle()).toBe('');
      expect(component.isGDE()).toBe(false);
      expect(component.status()).toBe('idle');
      expect(component.canSubmit()).toBe(false);
    });

    it('should render the submit button disabled', () => {
      expect(submitButton().disabled).toBe(true);
    });
  });

  describe('submit button blocking', () => {
    it('should stay disabled when the e-mail is invalid', () => {
      fillValidForm();
      typeInto('submission-email', 'not-an-email');

      expect(component.canSubmit()).toBe(false);
      expect(submitButton().disabled).toBe(true);
    });

    it('should stay disabled when the talk title is empty or whitespace', () => {
      fillValidForm();
      typeInto('submission-talk-title', '   ');

      expect(component.canSubmit()).toBe(false);
      expect(submitButton().disabled).toBe(true);
    });

    it('should be enabled when all fields are valid', () => {
      fillValidForm();

      expect(component.canSubmit()).toBe(true);
      expect(submitButton().disabled).toBe(false);
    });

    it('should be disabled again while the submission is in flight', () => {
      fillValidForm();
      submit();

      expect(component.status()).toBe('submitting');
      expect(submitButton().disabled).toBe(true);
      expect(form().getAttribute('aria-busy')).toBe('true');
      expect(liveRegion().textContent).toContain('Enviando…');

      const req = httpTesting.expectOne('/api/submissions');
      expect(req.request.body).toEqual(validSubmission);
      req.flush({ id: 'abc-123', ...validSubmission });
    });
  });

  describe('accessibility', () => {
    it('should link an invalid, touched field to its error message', () => {
      typeInto('submission-email', 'not-an-email');

      const email = input('submission-email');
      expect(email.getAttribute('aria-invalid')).toBe('true');
      const errorId = email.getAttribute('aria-describedby');
      expect(errorId).toBe('submission-email-error');
      expect(element.querySelector(`#${errorId}`)?.textContent).toContain(
        'e-mail válido'
      );
    });

    it('should not mark untouched fields as invalid', () => {
      const name = input('submission-name');
      expect(name.hasAttribute('aria-invalid')).toBe(false);
      expect(name.hasAttribute('aria-describedby')).toBe(false);
    });

    it('should expose a polite live region', () => {
      expect(liveRegion().getAttribute('aria-live')).toBe('polite');
    });
  });

  describe('server responses', () => {
    it('should announce success and reset the form on 201', () => {
      fillValidForm();
      submit();
      httpTesting
        .expectOne('/api/submissions')
        .flush({ id: 'abc-123', ...validSubmission });
      fixture.detectChanges();

      expect(component.status()).toBe('success');
      expect(liveRegion().textContent).toContain('enviada com sucesso');
      expect(component.name()).toBe('');
      expect(submitButton().disabled).toBe(true);
    });

    it('should set status to error and announce the 400 message', () => {
      fillValidForm();
      submit();
      httpTesting.expectOne('/api/submissions').flush(
        {
          statusCode: 400,
          error: 'Bad Request',
          message: ['email must be an email'],
        },
        { status: 400, statusText: 'Bad Request' }
      );
      fixture.detectChanges();

      expect(component.status()).toBe('error');
      expect(liveRegion().textContent).toContain('email must be an email');
      expect(submitButton().disabled).toBe(false);
    });
  });
});
