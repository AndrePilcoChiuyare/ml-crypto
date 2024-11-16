import { PredictionHistorical } from './../models/prediction-historical.model';
import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
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

  getDateRange(): Observable<string> {
    return this.http.get(`${this.baseUrl}/last-date/meme`, { responseType: 'text' });
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

  getDataAndPredict(): Observable<string> {
    // the model is prediction-request
    const body = {
      days_to_predict: 7,
      model: 'catboost'
    };
    return this.http.post(`${this.baseUrl}/get-data/predict-all`, body, { responseType: 'text' });
  } 

  getPredictionById(category: string, tokenId: string): Observable<PredictionHistorical> {
    return this.http.get<PredictionHistorical>(`${this.baseUrl}/categories/${category}/tokens/${tokenId}`);
  }

  postPrediction(data: any): Observable<any> {
    return this.http.post(`${this.baseUrl}/predict`, data);
  }
}
