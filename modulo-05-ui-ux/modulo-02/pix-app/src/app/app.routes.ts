import { Routes } from '@angular/router';

const loadPixTransfer = () =>
  import('./pix-transfer/pix-transfer').then((m) => m.PixTransferComponent);

export const routes: Routes = [
  { path: '', loadComponent: loadPixTransfer, pathMatch: 'full' },
  { path: 'pix', loadComponent: loadPixTransfer },
  { path: '**', redirectTo: '' },
];
