import { ValidationPipe } from '@nestjs/common';

/**
 * Global validation used by main.ts and by controller tests, so tests exercise
 * the exact production configuration.
 */
export function createValidationPipe(): ValidationPipe {
  return new ValidationPipe({
    whitelist: true,
    forbidNonWhitelisted: true,
    transform: true,
    transformOptions: { enableImplicitConversion: false },
  });
}
