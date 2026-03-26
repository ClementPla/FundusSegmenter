import { Component, OnInit, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { Router } from '@angular/router';
import { ImagesService } from '../../Services/images.service';

const PAGE_SIZE = 16;

type TileState = 'pending' | 'loading' | 'ready' | 'error';

@Component({
  selector: 'app-gallery',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    MatProgressSpinnerModule,
  ],
  templateUrl: './gallery.component.html',
  styleUrl: './gallery.component.scss',
})
export class GalleryComponent implements OnInit {
  currentPage = 0;
  private tileState = new Map<string, TileState>();

  constructor(
    public images: ImagesService,
    private ngZone: NgZone,
    private router: Router,
  ) {}

  // ── derived ────────────────────────────────────────────────────────────

  get totalPages(): number {
    return Math.max(1, Math.ceil(this.images.files.length / PAGE_SIZE));
  }

  get pageFiles(): string[] {
    const start = this.currentPage * PAGE_SIZE;
    return this.images.files.slice(start, start + PAGE_SIZE);
  }

  get pageIndices(): number[] {
    return Array.from({ length: this.totalPages }, (_, i) => i);
  }

  // ── lifecycle ──────────────────────────────────────────────────────────

  ngOnInit(): void {
    this.loadCurrentPage();
  }

  // ── pagination ─────────────────────────────────────────────────────────

  goToPage(page: number): void {
    if (page === this.currentPage) return;
    this.currentPage = page;
    this.loadCurrentPage();
  }

  // ── tile helpers ───────────────────────────────────────────────────────

  tileStatus(path: string): TileState {
    return this.tileState.get(path) ?? 'pending';
  }

  src(path: string): string | null {
    return this.images.getCachedSrc(path);
  }

  // ── actions ────────────────────────────────────────────────────────────

  activateImage(path: string): void {
    this.images.activeImage = path;
    this.router.navigate(['/editor']);
  }

  // ── private ────────────────────────────────────────────────────────────

  private loadCurrentPage(): void {
    for (const path of this.pageFiles) {
      const state = this.tileState.get(path);
      if (state === 'ready' || state === 'loading') continue;

      this.tileState.set(path, 'loading');
      this.images
        .loadImageSrc(path)
        .then(() => {
          this.ngZone.run(() => this.tileState.set(path, 'ready'));
        })
        .catch(() => {
          this.ngZone.run(() => this.tileState.set(path, 'error'));
        });
    }
  }
}
