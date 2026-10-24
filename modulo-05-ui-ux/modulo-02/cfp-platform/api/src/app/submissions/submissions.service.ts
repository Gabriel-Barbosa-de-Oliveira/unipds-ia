import { Injectable } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { CreateSpeakerSubmission, SpeakerDTO } from 'shared-types';

@Injectable()
export class SubmissionsService {
  // In-memory store: persistence is out of scope for this change.
  private readonly submissions: SpeakerDTO[] = [];

  create(input: CreateSpeakerSubmission): SpeakerDTO {
    const submission: SpeakerDTO = {
      id: randomUUID(),
      name: input.name,
      email: input.email,
      talkTitle: input.talkTitle,
      isGDE: input.isGDE,
    };
    this.submissions.push(submission);
    return submission;
  }
}
