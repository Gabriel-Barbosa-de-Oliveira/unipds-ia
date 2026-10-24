import { Body, Controller, Post } from '@nestjs/common';
import { SpeakerDTO } from 'shared-types';
import { CreateSubmissionBody } from './create-submission.body';
import { SubmissionsService } from './submissions.service';

@Controller('submissions')
export class SubmissionsController {
  constructor(private readonly submissionsService: SubmissionsService) {}

  @Post()
  create(@Body() body: CreateSubmissionBody): SpeakerDTO {
    return this.submissionsService.create(body);
  }
}
