import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { CreateEvent, EventDTO } from 'shared-types';
import { EventService } from './event.service';

describe('EventService', () => {
  let service: EventService;
  let httpTesting: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(EventService);
    httpTesting = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpTesting.verify());

  it('should POST the event to /api/events and return the EventDTO', () => {
    const event: CreateEvent = {
      name: 'Angular Summit',
      address: 'Av. Paulista, 1000',
      capacity: 200,
      date: '2027-03-15',
    };
    const created: EventDTO = { id: 'abc-123', ...event };
    let response: EventDTO | undefined;

    service.create(event).subscribe((value) => (response = value));

    const req = httpTesting.expectOne('/api/events');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(event);
    req.flush(created);

    expect(response).toEqual(created);
  });
});
