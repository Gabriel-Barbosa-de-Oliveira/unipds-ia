export interface EventDTO {
  id: string;
  name: string;
  address: string;
  capacity: number;
  /** Calendar date in `YYYY-MM-DD` format. */
  date: string;
}

export type CreateEvent = Omit<EventDTO, 'id'>;
