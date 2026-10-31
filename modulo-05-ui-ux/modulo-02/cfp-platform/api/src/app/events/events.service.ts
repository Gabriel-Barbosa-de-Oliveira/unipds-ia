import { Injectable } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { CreateEvent, EventDTO } from 'shared-types';

@Injectable()
export class EventsService {
  // In-memory store: persistence is out of scope for this change.
  private readonly events: EventDTO[] = [];

  create(input: CreateEvent): EventDTO {
    const event: EventDTO = {
      id: randomUUID(),
      name: input.name,
      address: input.address,
      capacity: input.capacity,
      date: input.date,
    };
    this.events.push(event);
    return event;
  }
}
