import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'scientificFormat',
  standalone: true
})
export class ScientificFormatPipe implements PipeTransform {

  transform(value: number): string {
    // If the number is less than 0.000001, use scientific notation with 6 decimal places
    if (value < 0.000001) {
      return value.toExponential(6);
    }

    // Otherwise, format the number with up to 6 decimal places
    return value.toFixed(6);
  }

}