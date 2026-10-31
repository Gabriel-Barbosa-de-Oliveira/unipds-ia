export interface SpeakerDTO {
  id: string;
  name: string;
  email: string;
  talkTitle: string;
  isGDE: boolean;
}

export type CreateSpeakerSubmission = Omit<SpeakerDTO, 'id'>;
