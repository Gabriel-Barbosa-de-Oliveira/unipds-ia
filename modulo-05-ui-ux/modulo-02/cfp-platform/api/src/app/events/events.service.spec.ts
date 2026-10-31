import { CreateEvent } from 'shared-types';
import { EventsService } from './events.service';

describe('EventsService', () => {
  const input: CreateEvent = {
    name: 'Angular Summit',
    address: 'Av. Paulista, 1000',
    capacity: 200,
    date: '2027-03-15',
  };

  let service: EventsService;

  beforeEach(() => {
    service = new EventsService();
  });

  it('generates an id', () => {
    const result = service.create(input);
    expect(typeof result.id).toBe('string');
    expect(result.id.length).toBeGreaterThan(0);
  });

  it('generates a unique id per event', () => {
    const first = service.create(input);
    const second = service.create(input);
    expect(first.id).not.toEqual(second.id);
  });

  it('preserves the submitted fields', () => {
    const { id, ...rest } = service.create(input);
    expect(id).toBeDefined();
    expect(rest).toEqual(input);
  });
});
