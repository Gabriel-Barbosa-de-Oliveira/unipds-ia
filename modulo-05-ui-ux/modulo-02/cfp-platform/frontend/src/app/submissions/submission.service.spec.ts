import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { CreateSpeakerSubmission, SpeakerDTO } from 'shared-types';
import { SubmissionService } from './submission.service';

describe('SubmissionService', () => {
  let service: SubmissionService;
  let httpTesting: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(SubmissionService);
    httpTesting = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpTesting.verify());

  it('should POST the submission to /api/submissions and return the SpeakerDTO', () => {
    const submission: CreateSpeakerSubmission = {
      name: 'Ada Lovelace',
      email: 'ada@example.com',
      talkTitle: 'Signals in practice',
      isGDE: false,
    };
    const created: SpeakerDTO = { id: 'abc-123', ...submission };
    let response: SpeakerDTO | undefined;

    service.submit(submission).subscribe((value) => (response = value));

    const req = httpTesting.expectOne('/api/submissions');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(submission);
    req.flush(created);

    expect(response).toEqual(created);
  });
});
