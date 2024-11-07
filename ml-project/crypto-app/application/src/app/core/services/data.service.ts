import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private baseUrl = 'http://localhost:5000'; // Update if needed

  constructor(private http: HttpClient) {}

  getPredictionsComplete(category: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/predictionsComplete/${category}`);
  }

  getPredictionsBasic(category: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/predictions-basic/${category}`);
  }

  getPredictionById(category: string, tokenId: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/predictions`, { category, id: tokenId });
  }

  postPrediction(data: any): Observable<any> {
    return this.http.post(`${this.baseUrl}/predict`, data);
  }
}
