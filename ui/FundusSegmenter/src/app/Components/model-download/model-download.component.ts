import { Component, OnInit, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { ModelService } from '../../Services/model.service';

@Component({
  selector: 'app-model-download',
  standalone: true,
  imports: [CommonModule, MatIconModule],
  templateUrl: './model-download.component.html',
  styleUrl: './model-download.component.scss',
})
export class ModelDownloadComponent implements OnInit {
  @Output() completed = new EventEmitter<void>();

  constructor(public model: ModelService) {}

  ngOnInit(): void {
    // Auto-start check
    this.model.checkModel().then((exists) => {
      if (exists) this.completed.emit();
    });
  }

  async startDownload(): Promise<void> {
    await this.model.downloadModel();
    if (this.model.status === 'ready') {
      // Small delay so the user sees "Complete" before dismissing
      setTimeout(() => this.completed.emit(), 600);
    }
  }

  async retry(): Promise<void> {
    await this.model.redownloadModel();
    if (this.model.status === 'ready') {
      setTimeout(() => this.completed.emit(), 600);
    }
  }
}
