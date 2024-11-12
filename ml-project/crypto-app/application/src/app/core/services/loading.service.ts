import { Injectable } from '@angular/core';
import { BehaviorSubject, finalize, Subject } from 'rxjs';
import { DataService } from './data.service';

@Injectable({
  providedIn: 'root',
})
export class LoadingService {
  private loadingSubject = new BehaviorSubject<boolean>(false);
  public isLoading$ = this.loadingSubject.asObservable();
  private fetchDataSubject = new Subject<void>();

  // Observable to trigger fetchData
  fetchData$ = this.fetchDataSubject.asObservable();

  constructor(private dataService: DataService) {}

  setLoading(isLoading: boolean) {
    this.loadingSubject.next(isLoading);
  }

  triggerFetchData(): void {
    this.fetchDataSubject.next();
  }

}