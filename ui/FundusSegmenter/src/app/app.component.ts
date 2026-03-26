import { Component, OnInit } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { MatIconModule } from '@angular/material/icon';
import { ModelDownloadComponent } from './Components/model-download/model-download.component';
import { ModelService } from './Services/model.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterOutlet,
    RouterLink,
    RouterLinkActive,
    MatIconModule,
    ModelDownloadComponent,
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent implements OnInit {
  title = 'FundusSegmenter';
  showModelDownload = true; // start visible; dismissed once model is confirmed

  constructor(private modelService: ModelService) {}

  async ngOnInit(): Promise<void> {
    // Quick check — if model already exists, dismiss immediately
    const exists = await this.modelService.checkModel();
    if (exists) {
      this.showModelDownload = false;
    }
  }

  onModelReady(): void {
    this.showModelDownload = false;
  }
}
