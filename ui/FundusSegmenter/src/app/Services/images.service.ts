import { Injectable, NgZone } from '@angular/core';
import { invoke } from '@tauri-apps/api/core';
import { readDir } from '@tauri-apps/plugin-fs';

const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg'];

export function isImagePath(path: string): boolean {
  return IMAGE_EXTENSIONS.some((ext) => path.toLowerCase().endsWith(ext));
}

async function collectImagesRecursively(dir: string): Promise<string[]> {
  const entries = await readDir(dir);
  const results: string[] = [];

  for (const entry of entries) {
    const fullPath = `${dir}/${entry.name}`;
    if (entry.isDirectory) {
      // It's a directory — recurse
      const nested = await collectImagesRecursively(fullPath);
      results.push(...nested);
    } else if (isImagePath(entry.name)) {
      results.push(fullPath);
    }
  }

  return results;
}

@Injectable({ providedIn: 'root' })
export class ImagesService {
  files: string[] = [];

  previewPath: string | null = null;
  previewSrc: string | null = null;
  previewLoading = false;
  activeImage: string | null = null;

  private cache = new Map<string, string>();

  constructor(private ngZone: NgZone) {}

  async loadImageSrc(path: string): Promise<string> {
    if (this.cache.has(path)) return this.cache.get(path)!;
    const src = await invoke<string>('read_image_base64', { path });
    this.cache.set(path, src);
    return src;
  }

  getCachedSrc(path: string): string | null {
    return this.cache.get(path) ?? null;
  }

  addFiles(paths: string[]): void {
    const existing = new Set(this.files);
    for (const p of paths) {
      if (!existing.has(p)) {
        this.files.push(p);
        existing.add(p);
      }
    }
  }

  async addFolder(folder: string): Promise<void> {
    const paths = await collectImagesRecursively(folder);
    this.ngZone.run(() => this.addFiles(paths));
  }

  removeFile(index: number): void {
    const removed = this.files.splice(index, 1)[0];
    this.cache.delete(removed);
    if (this.previewPath === removed) {
      this.previewPath = null;
      this.previewSrc = null;
    }
  }

  clearFiles(): void {
    this.files = [];
    this.cache.clear();
    this.previewPath = null;
    this.previewSrc = null;
    this.previewLoading = false;
  }

  async hoverPreview(path: string): Promise<void> {
    this.previewPath = path;

    if (this.cache.has(path)) {
      this.previewSrc = this.cache.get(path)!;
      return;
    }

    this.previewLoading = true;
    this.previewSrc = null;

    try {
      const src = await invoke<string>('read_image_base64', { path });
      this.cache.set(path, src);
      if (this.previewPath === path) {
        this.ngZone.run(() => {
          this.previewSrc = src;
          this.previewLoading = false;
        });
      }
    } catch (err) {
      console.error('Failed to load preview:', err);
      this.ngZone.run(() => (this.previewLoading = false));
    }
  }

  clearPreview(): void {
    this.previewPath = null;
    this.previewSrc = null;
    this.previewLoading = false;
  }

  basename(path: string): string {
    return path.split(/[\\/]/).pop() ?? path;
  }
}
