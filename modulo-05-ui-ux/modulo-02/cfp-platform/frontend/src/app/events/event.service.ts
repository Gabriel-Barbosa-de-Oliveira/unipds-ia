import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { CreateEvent, EventDTO } from 'shared-types';

@Injectable({ providedIn: 'root' })
export class EventService {
  private readonly http = inject(HttpClient);

  create(event: CreateEvent): Observable<EventDTO> {
    return this.http.post<EventDTO>('/api/events', event);
  }
}
