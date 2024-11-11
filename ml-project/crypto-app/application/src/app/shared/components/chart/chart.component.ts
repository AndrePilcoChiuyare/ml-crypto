import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import {
  ApexAxisChartSeries,
  ApexChart,
  ApexXAxis,
  ApexTitleSubtitle,
  NgApexchartsModule,
  ApexAnnotations,
  ApexTooltip,
  ApexYAxis,
  ApexFill
} from 'ng-apexcharts';
import { DataService } from '../../../core/services/data.service';

export type ChartOptions = {
  series: ApexAxisChartSeries;
  chart: ApexChart;
  xaxis: ApexXAxis;
  yaxis: ApexYAxis;
  title: ApexTitleSubtitle;
  annotations: ApexAnnotations;
  tooltip: ApexTooltip;
  fill: ApexFill;
};

@Component({
  selector: 'app-chart',
  standalone: true,
  imports: [CommonModule, NgApexchartsModule],
  templateUrl: './chart.component.html',
  styleUrls: ['./chart.component.css']
})
export class ChartComponent {
  @Input() category!: string;
  @Input() tokenId!: string;
  @Input() tokenName: string = '';
  public seriesData: { x: Date, y: number }[] = [];

  public chartOptions!: Partial<ChartOptions>;

  constructor(private dataService: DataService) {}

  ngOnInit(): void {
    this.fetchHistoricalData();
  }

  fetchHistoricalData(): void {
    this.dataService.getPredictionById(this.category, this.tokenId).subscribe((response) => {
      const prediction = response;
      this.seriesData = prediction.close_data.map(data => ({
        x: new Date(data.timestamp),
        y: data.close
      }));

      this.chartOptions = {
        series: [
          {
            name: 'Close Price',
            data: this.seriesData
          }
        ],
        chart: {
          type: 'line',
          height: 350,
          zoom: {
            enabled: true,
            type: 'x', // Enable zoom on x-axis
            autoScaleYaxis: true, // Automatically scales the y-axis
          },
          events: {
            zoomed: (chartContext, { xaxis }) => {
              const filteredData = this.seriesData.filter(
                (dataPoint) => dataPoint.x.getTime() >= xaxis.min && dataPoint.x.getTime() <= xaxis.max
              );
              if (filteredData.length > 0) {
                this.chartOptions.yaxis = {
                  min: Math.min(...filteredData.map(data => data.y)),
                  max: Math.max(...filteredData.map(data => data.y)),
                  title: {
                    text: 'Close Price'
                  }
                };
              }
            },
          }
        },
        annotations: {
          xaxis: [
            {
              x: this.seriesData[this.seriesData.length - 7].x.getTime(),
              x2: this.seriesData[this.seriesData.length - 1].x.getTime(),
              fillColor: '#775DD0',
              label: {
                text: 'Predicted days',
              }
            },
            {
              x: new Date('28 Nov 2012').getTime(),
              borderColor: '#775DD0',
              label: {
                style: {
                  color: '#775DD0',
                },
                text: 'First halving'
              }
            },
            {
              x: new Date('09 Jul 2016').getTime(),
              borderColor: '#775DD0',
              label: {
                style: {
                  color: '#775DD0',
                },
                text: 'Second halving'
              }
            },
            {
              x: new Date('11 May 2020').getTime(),
              borderColor: '#775DD0',
              label: {
                style: {
                  color: '#775DD0',
                },
                text: 'Third halving'
              }
            },
            {
              x: new Date('20 Apr 2024').getTime(),
              borderColor: '#775DD0',
              label: {
                style: {
                  color: '#775DD0',
                },
                text: 'Fourth halving'
              }
            }
          ]
        },
        tooltip: {
          x: {
            format: "dd MMM yyyy"
          }
        },
        xaxis: {
          type: 'datetime',
          title: {
            text: 'Date'
          }
        },
        yaxis: {
          title: {
            text: 'Close Price'
          }
        },
        title: {
          text: `Close Price History for ${prediction.name}`,
          align: 'left'
        },
        fill: {
          type: "gradient",
          gradient: {
            shadeIntensity: 1,
            opacityFrom: 0.7,
            opacityTo: 0.9,
            stops: [0, 100]
          }
        }
      };
    });
  }
}
