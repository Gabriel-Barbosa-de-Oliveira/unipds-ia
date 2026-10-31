import { INestApplication } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import request from 'supertest';
import { CreateEvent } from 'shared-types';
import { createValidationPipe } from '../validation.pipe';
import { EventsModule } from './events.module';
import { EventsService } from './events.service';

describe('EventsController (POST /events)', () => {
  const valid: CreateEvent = {
    name: 'Angular Summit',
    address: 'Av. Paulista, 1000',
    capacity: 200,
    date: '2027-03-15',
  };

  let app: INestApplication;
  let createSpy: jest.SpyInstance;

  beforeAll(async () => {
    const moduleRef = await Test.createTestingModule({
      imports: [EventsModule],
    }).compile();

    app = moduleRef.createNestApplication();
    app.useGlobalPipes(createValidationPipe());
    await app.init();
    createSpy = jest.spyOn(app.get(EventsService), 'create');
  });

  beforeEach(() => {
    createSpy.mockClear();
  });

  afterAll(async () => {
    await app.close();
  });

  const post = (body: unknown) =>
    request(app.getHttpServer()).post('/events').send(body as object);

  describe('accepts valid payloads with 201 Created', () => {
    it('returns the event with a generated id', async () => {
      const response = await post(valid).expect(201);
      expect(response.body).toEqual({ ...valid, id: expect.any(String) });
      expect(response.body.id.length).toBeGreaterThan(0);
    });

    it('trims name and address', async () => {
      const response = await post({
        ...valid,
        name: '  Angular Summit  ',
        address: '  Av. Paulista, 1000  ',
      }).expect(201);
      expect(response.body.name).toBe(valid.name);
      expect(response.body.address).toBe(valid.address);
    });

    it('accepts name and address at their maximum length', () =>
      post({
        ...valid,
        name: 'a'.repeat(120),
        address: 'a'.repeat(200),
      }).expect(201));

    it('accepts a past date', () =>
      post({ ...valid, date: '2020-01-01' }).expect(201));
  });

  describe('rejects invalid payloads with 400 Bad Request', () => {
    afterEach(() => {
      expect(createSpy).not.toHaveBeenCalled();
    });

    it('empty body', () => post({}).expect(400));

    it.each(['name', 'address', 'capacity', 'date'] as const)(
      'missing %s',
      (field) => {
        const body: Record<string, unknown> = { ...valid };
        delete body[field];
        return post(body).expect(400);
      },
    );

    it.each([
      ['name', ''],
      ['name', '   '],
      ['address', ''],
      ['address', '   '],
    ])('%s empty or whitespace only (%j)', (field, value) =>
      post({ ...valid, [field]: value }).expect(400),
    );

    it('name longer than 120 characters', () =>
      post({ ...valid, name: 'a'.repeat(121) }).expect(400));

    it('address longer than 200 characters', () =>
      post({ ...valid, address: 'a'.repeat(201) }).expect(400));

    it.each([[0], [-1], [1.5], ['200']])('capacity %j', (capacity) =>
      post({ ...valid, capacity }).expect(400),
    );

    it.each([
      ['2027-02-30'],
      ['2027-13-01'],
      ['15/03/2027'],
      ['2027-03-15T10:00:00Z'],
      ['not-a-date'],
    ])('date %s', (date) => post({ ...valid, date }).expect(400));

    it.each([
      ['id', '123'],
      ['foo', 'bar'],
    ])('extra field %s', (field, value) =>
      post({ ...valid, [field]: value }).expect(400),
    );
  });
});
