import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { CreateSpeakerSubmission, SpeakerDTO } from 'shared-types';

@Injectable({ providedIn: 'root' })
export class SubmissionService {
  private readonly http = inject(HttpClient);

  submit(submission: CreateSpeakerSubmission): Observable<SpeakerDTO> {
    return this.http.post<SpeakerDTO>('/api/submissions', submission);
  }
}
