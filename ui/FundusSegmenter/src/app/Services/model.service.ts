import { Injectable, NgZone } from '@angular/core';
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

export type ModelStatus =
  | 'checking'
  | 'ready'
  | 'missing'
  | 'downloading'
  | 'error';

export interface DownloadProgress {
  downloaded: number;
  total: number | null;
  fraction: number;
  status: string;
}

/**
 * The HuggingFace URL for the ONNX model.
 * Update this when the model changes.
 */
const MODEL_URL =
  'https://huggingface.co/ClementP/FundusSegmenter/resolve/main/ensemble_segmenter.onnx?download=true';

@Injectable({ providedIn: 'root' })
export class ModelService {
  status: ModelStatus = 'checking';
  progress: DownloadProgress = {
    downloaded: 0,
    total: null,
    fraction: 0,
    status: '',
  };
  error: string | null = null;

  private unlisten?: UnlistenFn;

  constructor(private ngZone: NgZone) {}

  /** Check if the model exists on disk. Call once at app startup. */
  async checkModel(): Promise<boolean> {
    this.status = 'checking';
    this.error = null;
    try {
      const exists = await invoke<boolean>('check_model_exists');
      this.ngZone.run(() => {
        this.status = exists ? 'ready' : 'missing';
      });
      return exists;
    } catch (e) {
      this.ngZone.run(() => {
        this.status = 'error';
        this.error = `Failed to check model: ${e}`;
      });
      return false;
    }
  }

  /** Start downloading the model. Subscribes to progress events. */
  async downloadModel(url?: string): Promise<void> {
    this.status = 'downloading';
    this.error = null;
    this.progress = {
      downloaded: 0,
      total: null,
      fraction: 0,
      status: 'Starting…',
    };

    // Listen for progress events from Rust
    this.unlisten = await listen<DownloadProgress>(
      'model-download-progress',
      (event) => {
        this.ngZone.run(() => {
          this.progress = event.payload;
        });
      },
    );

    try {
      await invoke('download_model', { url: url ?? MODEL_URL });
      this.ngZone.run(() => {
        this.status = 'ready';
      });
    } catch (e) {
      this.ngZone.run(() => {
        this.status = 'error';
        this.error = `Download failed: ${e}`;
      });
    } finally {
      this.unlisten?.();
      this.unlisten = undefined;
    }
  }

  /** Delete and re-download (for model updates). */
  async redownloadModel(): Promise<void> {
    try {
      await invoke('delete_model');
    } catch (e) {
      console.warn('Could not delete existing model:', e);
    }
    await this.downloadModel();
  }

  /** Format bytes for display. */
  formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024)
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }
}
