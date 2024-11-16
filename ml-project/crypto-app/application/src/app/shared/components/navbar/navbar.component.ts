import { MatToolbarModule } from '@angular/material/toolbar';
import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { DataService } from '../../../core/services/data.service';
import { LoadingService } from '../../../core/services/loading.service';
import { MatButtonModule } from '@angular/material/button';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [MatToolbarModule, MatButtonModule, CommonModule],
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.css'
})
export class NavbarComponent implements OnInit{
  @Output() loading = new EventEmitter<boolean>();
  startDate: Date = new Date();
  endDate: Date = new Date();
  
  constructor(
    private dataService: DataService,
    private loadingService: LoadingService,
  ) {}

  ngOnInit(): void {
    this.getDate();
  }

  getDate():void {
    this.dataService.getDateRange().subscribe({
      next: (response) => {
        const [year, month, day] = response.split('-');
        this.startDate = new Date(+year, +month - 1, +day);
        this.endDate = new Date(+year, +month - 1, +day);
      },
    });
  }

  makePrediction(): void {
    this.loadingService.setLoading(true);
    this.dataService.getDataAndPredict().subscribe({
      next: (response) => {
        if (response === 'Predictions completed') {
          // Signal HomePageComponent to fetch data
          this.loadingService.triggerFetchData();
        }
      },
      error: () => this.loadingService.setLoading(false),
      complete: () => {
        // End loading when the prediction is completed
        this.getDate();
        this.loadingService.setLoading(false);
      },
    });
  }
}
