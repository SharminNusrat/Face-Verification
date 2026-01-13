import { Routes } from '@angular/router';
import { FaceVerificationComponent } from './features/face-verification/face-verification.component';

export const routes: Routes = [
    { path: '', redirectTo: 'verify', pathMatch: 'full' },
    { path: 'verify', component: FaceVerificationComponent }
];
