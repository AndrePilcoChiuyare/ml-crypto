import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { ChartComponent } from '../chart/chart.component';
import { DataService } from '../../../core/services/data.service';

@Component({
  selector: 'app-dialog',
  standalone: true,
  imports: [ChartComponent],
  templateUrl: './dialog.component.html',
  styleUrl: './dialog.component.css'
})
export class DialogComponent {
  
  constructor(
    @Inject(MAT_DIALOG_DATA) public data: any,
    private dataService: DataService,
  ) {}

  tokenID = '';
  category = '';
  seriesData: any;

  ngOnInit(): void {
    this.tokenID = this.data.id;
    this.category = this.data.category;
    console.log('tokenID:', this.tokenID);
    console.log('category:', this.category);
    this.fetchPredictionData();
  }

  fetchPredictionData(): void {
    this.dataService.getPredictionById(this.category, this.tokenID).subscribe((response) => {
      const prediction = response;
      this.seriesData = prediction.close_data.map(data => ({
        x: new Date(data.timestamp),
        y: data.close
      }));
    });
  }
}
