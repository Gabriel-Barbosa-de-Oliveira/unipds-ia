import { Routes } from '@angular/router';

const loadPixTransfer = () =>
  import('./pix-transfer/pix-transfer').then((m) => m.PixTransferComponent);

const loadPixHistory = () =>
  import('./pix-history/pix-history').then((m) => m.PixHistoryComponent);

export const routes: Routes = [
  { path: '', loadComponent: loadPixTransfer, pathMatch: 'full' },
  { path: 'pix', loadComponent: loadPixTransfer },
  { path: 'extrato', loadComponent: loadPixHistory },
  { path: '**', redirectTo: '' },
];
