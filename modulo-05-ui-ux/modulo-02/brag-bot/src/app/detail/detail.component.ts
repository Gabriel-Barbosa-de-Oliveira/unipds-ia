import { ChangeDetectionStrategy, Component, computed, inject, input } from '@angular/core';
import { RouterLink } from '@angular/router';
import { BragService } from '../services/brag.service';

@Component({
  selector: 'app-detail',
  imports: [RouterLink],
  templateUrl: './detail.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DetailComponent {
  private readonly bragService = inject(BragService);

  readonly id = input.required<string>();

  protected readonly brag = computed(() =>
    this.bragService.brags().find((brag) => brag.id === this.id()),
  );
}
