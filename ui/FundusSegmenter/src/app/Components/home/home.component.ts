import { Component, OnInit, OnDestroy, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatListModule } from '@angular/material/list';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { open } from '@tauri-apps/plugin-dialog';
import { getCurrentWebviewWindow } from '@tauri-apps/api/webviewWindow';
import type { UnlistenFn } from '@tauri-apps/api/event';
import { ImagesService, isImagePath } from '../../Services/images.service';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    CommonModule,
    MatListModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss',
})
export class HomeComponent implements OnInit, OnDestroy {
  isDragOver = false;
  private unlistenDragDrop?: UnlistenFn;

  constructor(
    private ngZone: NgZone,
    public images: ImagesService,
  ) {}

  async ngOnInit() {
    const win = getCurrentWebviewWindow();

    this.unlistenDragDrop = await win.onDragDropEvent((event) => {
      this.ngZone.run(async () => {
        if (event.payload.type === 'over') {
          this.isDragOver = true;
        } else if (event.payload.type === 'leave') {
          this.isDragOver = false;
        } else if (event.payload.type === 'drop') {
          this.isDragOver = false;
          const { paths } = event.payload;
          const imageFiles = paths.filter(isImagePath);
          const folders = paths.filter((p) => !isImagePath(p));

          this.images.addFiles(imageFiles);
          for (const folder of folders) {
            await this.images.addFolder(folder);
          }
        }
      });
    });
  }

  ngOnDestroy() {
    this.unlistenDragDrop?.();
  }

  async loadImages() {
    const selected = await open({
      multiple: true,
      filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg'] }],
    });
    if (selected) {
      this.ngZone.run(() =>
        this.images.addFiles(Array.isArray(selected) ? selected : [selected]),
      );
    }
  }

  async loadFolder() {
    const folder = await open({ directory: true });
    if (folder) await this.images.addFolder(folder as string);
  }
}
