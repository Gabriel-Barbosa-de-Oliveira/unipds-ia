import { Routes } from '@angular/router';

const pageTitle = 'Brag-Bot | Pós IA UNIPDS';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    title: pageTitle,
    loadComponent: () =>
      import('./dashboard/dashboard.component').then((m) => m.DashboardComponent),
  },
  {
    path: 'detail/:id',
    title: pageTitle,
    loadComponent: () => import('./detail/detail.component').then((m) => m.DetailComponent),
  },
  { path: '**', redirectTo: '' },
];
