import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { map, Observable } from 'rxjs';
import { PredictionBasic } from '../models/prediction-basic.model';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private baseUrl = 'http://localhost:5000'; // Update if needed

  constructor(private http: HttpClient) {}

  getPredictionsComplete(category: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/predictionsComplete/${category}`);
  }

  getPredictionsBasic(category: string): Observable<Record<string, PredictionBasic>> {
    return this.http.get<Record<string, PredictionBasic>>(`${this.baseUrl}/predictions-basic/${category}`).pipe(
      map(predictions => {
      const sortedPredictions = Object.entries(predictions)
        .sort(([, a], [, b]) => b.future_multiply - a.future_multiply)
        .reduce((acc, [key, value]) => {
        acc[key] = value;
        return acc;
        }, {} as Record<string, PredictionBasic>);
      return sortedPredictions;
      })
    );
  }

  getPredictionById(category: string, tokenId: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/predictions`, { category, id: tokenId });
  }

  postPrediction(data: any): Observable<any> {
    return this.http.post(`${this.baseUrl}/predict`, data);
  }
}
