import { INestApplication } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import request from 'supertest';
import { CreateSpeakerSubmission } from 'shared-types';
import { createValidationPipe } from '../validation.pipe';
import { SubmissionsModule } from './submissions.module';

describe('SubmissionsController (POST /submissions)', () => {
  const valid: CreateSpeakerSubmission = {
    name: 'Ada Lovelace',
    email: 'ada@example.com',
    talkTitle: 'Signals in Angular',
    isGDE: false,
  };

  let app: INestApplication;

  beforeAll(async () => {
    const moduleRef = await Test.createTestingModule({
      imports: [SubmissionsModule],
    }).compile();

    app = moduleRef.createNestApplication();
    app.useGlobalPipes(createValidationPipe());
    await app.init();
  });

  afterAll(async () => {
    await app.close();
  });

  const post = (body: unknown) =>
    request(app.getHttpServer()).post('/submissions').send(body as object);

  describe('rejects invalid payloads with 400 Bad Request', () => {
    it('empty body', () => post({}).expect(400));

    it.each(['name', 'email', 'talkTitle', 'isGDE'] as const)(
      'missing %s',
      (field) => {
        const body: Record<string, unknown> = { ...valid };
        delete body[field];
        return post(body).expect(400);
      },
    );

    it('invalid email', () => post({ ...valid, email: 'not-an-email' }).expect(400));

    it.each([
      ['name', ''],
      ['name', '   '],
      ['talkTitle', ''],
      ['talkTitle', '   '],
    ])('%s empty or whitespace only (%j)', (field, value) =>
      post({ ...valid, [field]: value }).expect(400),
    );

    it('name longer than 120 characters', () =>
      post({ ...valid, name: 'a'.repeat(121) }).expect(400));

    it('talkTitle longer than 200 characters', () =>
      post({ ...valid, talkTitle: 'a'.repeat(201) }).expect(400));

    it('isGDE sent as the string "true"', () =>
      post({ ...valid, isGDE: 'true' }).expect(400));

    it.each([
      ['id', '123'],
      ['foo', 'bar'],
    ])('extra field %s', (field, value) =>
      post({ ...valid, [field]: value }).expect(400),
    );
  });

  describe('accepts a valid payload', () => {
    it('returns 201 with a generated id and the submitted fields', async () => {
      const res = await post(valid).expect(201);
      const { id, ...rest } = res.body;
      expect(typeof id).toBe('string');
      expect(id.length).toBeGreaterThan(0);
      expect(rest).toEqual(valid);
    });

    it('accepts boundary lengths (120 / 200)', () =>
      post({ ...valid, name: 'a'.repeat(120), talkTitle: 'a'.repeat(200) }).expect(201));
  });
});

describe('SubmissionsController (GET /submissions)', () => {
  const ada: CreateSpeakerSubmission = {
    name: 'Ada Lovelace',
    email: 'ada@example.com',
    talkTitle: 'Signals in Angular',
    isGDE: true,
  };
  const grace: CreateSpeakerSubmission = {
    name: 'Grace Hopper',
    email: 'grace@example.com',
    talkTitle: 'Zoneless change detection',
    isGDE: false,
  };

  let app: INestApplication;

  // A fresh app per test keeps the in-memory store empty between tests.
  beforeEach(async () => {
    const moduleRef = await Test.createTestingModule({
      imports: [SubmissionsModule],
    }).compile();

    app = moduleRef.createNestApplication();
    app.useGlobalPipes(createValidationPipe());
    await app.init();
  });

  afterEach(async () => {
    await app.close();
  });

  const post = (body: unknown) =>
    request(app.getHttpServer()).post('/submissions').send(body as object);
  const list = () => request(app.getHttpServer()).get('/submissions');

  it('returns 200 with an empty array before any submission', () =>
    list().expect(200).expect([]));

  it('returns accepted submissions in submission order', async () => {
    const first = await post(ada).expect(201);
    const second = await post(grace).expect(201);

    const res = await list().expect(200);
    expect(res.body).toEqual([first.body, second.body]);
  });

  it('does not list rejected submissions', async () => {
    await post({ ...ada, email: 'not-an-email' }).expect(400);

    const res = await list().expect(200);
    expect(res.body).toEqual([]);
  });
});
