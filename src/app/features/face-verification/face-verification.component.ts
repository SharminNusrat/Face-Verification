import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/services/api.service';
import { VerificationResponse } from '../../core/models/verification.model';
import { finalize } from 'rxjs/operators';

@Component({
    selector: 'app-face-verification',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './face-verification.component.html',
    styleUrl: './face-verification.component.css'
})
export class FaceVerificationComponent {
    image1: File | null = null;
    image2: File | null = null;
    image1Preview: string | null = null;
    image2Preview: string | null = null;
    result: VerificationResponse | null = null;
    loading = false;
    error: string | null = null;

    constructor(private apiService: ApiService, private cdr: ChangeDetectorRef) { }

    onFileSelected(event: any, imageNumber: number) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e: any) => {
                if (imageNumber === 1) {
                    this.image1 = file;
                    this.image1Preview = e.target.result;
                } else {
                    this.image2 = file;
                    this.image2Preview = e.target.result;
                }
                this.cdr.detectChanges(); // Force update for preview
            };
            reader.readAsDataURL(file);
        }
    }

    verify() {
        if (!this.image1 || !this.image2) {
            this.error = 'Please select both images.';
            return;
        }

        this.loading = true;
        this.error = null;
        this.result = null;

        this.apiService.verifyFaces(this.image1, this.image2)
            .pipe(finalize(() => {
                console.log('Verification request completed (finalize).');
                this.loading = false;
                this.cdr.detectChanges(); // Force update when done
            }))
            .subscribe({
                next: (response) => {
                    console.log('Verification success:', response);
                    if (response.error) {
                        this.error = response.error;
                    }
                    this.result = response;
                },
                error: (err) => {
                    console.error('Verification error:', err);
                    if (err.status === 0) {
                        this.error = 'Unable to connect to the server (Status 0). Ensure backend is running and CORS is enabled.';
                    } else if (err.error && typeof err.error.detail === 'string') {
                        this.error = err.error.detail;
                    } else if (err.message) {
                        this.error = err.message;
                    } else {
                        this.error = 'Verification failed. Please check the backend connection.';
                    }
                }
            });
    }
}
