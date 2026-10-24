import { CreateSpeakerSubmission } from 'shared-types';
import { SubmissionsService } from './submissions.service';

describe('SubmissionsService', () => {
  const input: CreateSpeakerSubmission = {
    name: 'Ada Lovelace',
    email: 'ada@example.com',
    talkTitle: 'Signals in Angular',
    isGDE: true,
  };

  let service: SubmissionsService;

  beforeEach(() => {
    service = new SubmissionsService();
  });

  it('generates an id', () => {
    const result = service.create(input);
    expect(typeof result.id).toBe('string');
    expect(result.id.length).toBeGreaterThan(0);
  });

  it('generates a unique id per submission', () => {
    const first = service.create(input);
    const second = service.create(input);
    expect(first.id).not.toEqual(second.id);
  });

  it('preserves the submitted fields', () => {
    const { id, ...rest } = service.create(input);
    expect(id).toBeDefined();
    expect(rest).toEqual(input);
  });

  describe('findAll', () => {
    it('returns an empty list before any submission', () => {
      expect(service.findAll()).toEqual([]);
    });

    it('returns submissions in insertion order', () => {
      const first = service.create(input);
      const second = service.create({ ...input, talkTitle: 'Zoneless Angular' });
      expect(service.findAll()).toEqual([first, second]);
    });

    it('returns a copy that does not affect the store', () => {
      service.create(input);
      service.findAll().pop();
      expect(service.findAll()).toHaveLength(1);
    });
  });
});
