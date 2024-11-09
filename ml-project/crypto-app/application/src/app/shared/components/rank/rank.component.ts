import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { MatTableModule } from '@angular/material/table';
import { DialogComponent } from '../dialog/dialog.component';
import { MatDialog } from '@angular/material/dialog';

@Component({
  selector: 'app-rank',
  standalone: true,
  imports: [CommonModule, MatTableModule, MatIconModule],
  templateUrl: './rank.component.html',
  styleUrl: './rank.component.css'
})
export class RankComponent {
  displayedColumns: string[] = ['image', 'token', 'last_close', 'last_pred_close', 'future_multiply', 'marketcap','market_cap_level'];
  @Input() data: any[] = [];

  constructor(public dialog: MatDialog) {}

  openDialog(item: any): void {
    this.dialog.open(DialogComponent, {
      data: item,
      minWidth: '30%',
      maxWidth: '100%',
      width: '70%',
    });
  }
}
