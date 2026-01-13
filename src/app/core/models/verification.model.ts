export interface VerificationResponse {
    verified: boolean;
    distance: number;
    threshold: number;
    model: string;
    similarity_metric: string;
    time: number; // Backend returns 'time', not 'time_taken'
}
