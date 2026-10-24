import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { SubmissionsModule } from './submissions/submissions.module';

@Module({
  imports: [SubmissionsModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
