import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RankComponent } from '../../../shared/components/rank/rank.component';
import { DataService } from '../../../core/services/data.service';
import { PredictionBasic } from '../../../core/models/prediction-basic.model';

@Component({
  selector: 'app-home-page',
  standalone: true,
  imports: [CommonModule, RankComponent],
  templateUrl: './home-page.component.html',
  styleUrl: './home-page.component.css'
})
export class HomePageComponent {
  data: PredictionBasic[] = [];  // Use the new interface
  selectedCategory = 'ai';

  constructor(private dataService: DataService) {}

  ngOnInit(): void {
    this.fetchData();
  }

  fetchData(): void {
    console.log('Fetching data for category:', this.selectedCategory);  // Debug line
    this.dataService.getPredictionsBasic(this.selectedCategory).subscribe({
      next: (data: Record<string, PredictionBasic>) => {
        this.data = Object.values(data);
        console.log('Data:', this.data);  // Debug line
      },
      error: (error) => {
        console.error('Error:', error);  // Debug line
      }
    });
  }

  onCategoryChange(category: string): void {
    this.selectedCategory = category;
    console.log('Selected Category:', this.selectedCategory); 
    this.fetchData();
  }
}
