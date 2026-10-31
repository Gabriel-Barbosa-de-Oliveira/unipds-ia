import { Body, Controller, Post } from '@nestjs/common';
import { EventDTO } from 'shared-types';
import { CreateEventBody } from './create-event.body';
import { EventsService } from './events.service';

@Controller('events')
export class EventsController {
  constructor(private readonly eventsService: EventsService) {}

  @Post()
  create(@Body() body: CreateEventBody): EventDTO {
    return this.eventsService.create(body);
  }
}
