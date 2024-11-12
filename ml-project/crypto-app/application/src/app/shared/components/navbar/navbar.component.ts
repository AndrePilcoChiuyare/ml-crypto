import { MatToolbarModule } from '@angular/material/toolbar';
import { Component, EventEmitter, Output } from '@angular/core';
import { DataService } from '../../../core/services/data.service';
import { LoadingService } from '../../../core/services/loading.service';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [MatToolbarModule],
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.css'
})
export class NavbarComponent {
  @Output() loading = new EventEmitter<boolean>();
  
  constructor(
    private dataService: DataService,
    private loadingService: LoadingService,
  ) {}

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
        this.loadingService.setLoading(false);
      },
    });
  }
}
