import { Route } from '@angular/router';

export const appRoutes: Route[] = [
  {
    path: '',
    pathMatch: 'full',
    title: 'Enviar proposta de palestra',
    loadComponent: () =>
      import('./submissions/submission-form.component').then(
        (m) => m.SubmissionFormComponent
      ),
  },
  {
    path: 'dashboard',
    title: 'Propostas enviadas',
    loadComponent: () =>
      import('./submissions/cfp-dashboard.component').then(
        (m) => m.CfpDashboardComponent
      ),
  },
];
