import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { VerificationResponse } from '../models/verification.model';

@Injectable({
    providedIn: 'root'
})
export class ApiService {
    private apiUrl = 'http://localhost:8000/api/v1/face/verify'; // Direct backend URL prefix

    constructor(private http: HttpClient) { }

    verifyFaces(image1: File, image2: File): Observable<VerificationResponse> {
        const formData = new FormData();
        formData.append('file1', image1);
        formData.append('file2', image2);

        return this.http.post<VerificationResponse>(this.apiUrl, formData);
    }
}
