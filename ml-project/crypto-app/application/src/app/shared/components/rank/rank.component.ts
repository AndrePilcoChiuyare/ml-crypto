import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-rank',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './rank.component.html',
  styleUrl: './rank.component.css'
})
export class RankComponent {
  @Input() data: any[] = [];
}
