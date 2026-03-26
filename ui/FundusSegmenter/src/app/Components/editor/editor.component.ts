import {
  Component,
  OnInit,
  ViewChild,
  ElementRef,
  AfterViewInit,
  NgZone,
  OnDestroy,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { ImagesService } from '../../Services/images.service';
import { Router } from '@angular/router';
import { invoke } from '@tauri-apps/api/core';
import {
  CLASS_STYLES,
  ClassPolygons,
  CLASS_NAMES,
  SegPolygon,
} from './segmentation.types';

import { polygonCentroid, polygonBBox } from './utils/geometry';
import {
  RegionCircle,
  RegionSummary,
  RegionName,
  REGION_ORDER,
  computeRegionCircles,
  computeRegionSummaries,
} from './utils/regions';
import {
  FeatureEntry,
  LESION_IDS,
  polyKey,
  assignPolygonRegions,
  computeFeatureVector,
} from './utils/features';
import {
  Prediction,
  loadModel,
  predict,
  getModel,
  loadScoreModel,
  predictScore,
  getScoreModel,
} from './utils/classifier';

const MIN_SCALE = 0.05;
const MAX_SCALE = 50;
const ZOOM_SENSITIVITY = 0.001;
const CLICK_THRESHOLD = 3; // px — above this distance a mousedown→mouseup is a pan, not a click

const OD_CLASS = 5;
const MACULA_CLASS = 6;

interface Transform {
  x: number;
  y: number;
  scale: number;
}

@Component({
  selector: 'app-editor',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule, MatTooltipModule],
  templateUrl: './editor.component.html',
  styleUrl: './editor.component.scss',
})
export class EditorComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('viewport') viewportRef!: ElementRef<HTMLDivElement>;
  @ViewChild('image') set imageContent(content: ElementRef<HTMLImageElement>) {
    if (content) {
      this.imageRef = content;
      setTimeout(() => {
        if (this.imageRef?.nativeElement?.complete) this.onImageLoad();
      }, 0);
    }
  }
  private imageRef!: ElementRef<HTMLImageElement>;

  protected readonly CLASS_NAMES = CLASS_NAMES;
  protected readonly CLASS_STYLES = CLASS_STYLES;
  protected readonly REGION_ORDER = REGION_ORDER;
  protected readonly Math = Math;

  // ── image state ─────────────────────────────────────────────────────
  src: string | null = null;
  imagePath: string | null = null;
  imageNaturalWidth = 0;
  imageNaturalHeight = 0;

  // ── transform ───────────────────────────────────────────────────────
  transform: Transform = { x: 0, y: 0, scale: 1 };
  isPanning = false;
  private panStart = { x: 0, y: 0 };
  private transformAtPanStart: Transform = { x: 0, y: 0, scale: 1 };
  private panMoved = false; // true if mouse moved > CLICK_THRESHOLD during pan

  // ── segmentation ────────────────────────────────────────────────────
  segmentation: ClassPolygons[] = [];
  ROI: SegPolygon | null = null;
  isProcessing = false;
  overlayOpacity = 1.0;
  fov = 45; // degrees — field of view of the fundus camera
  visibleClasses = new Set<number>();

  // ── landmarks ───────────────────────────────────────────────────────
  private detectedOdCenter: [number, number] | null = null;
  private detectedMacCenter: [number, number] | null = null;
  private detectedOdDiameter = 150;
  maculaOverride: [number, number] | null = null;
  isDraggingMacula = false;

  get odCenter(): [number, number] | null {
    return this.detectedOdCenter;
  }
  get macCenter(): [number, number] | null {
    return this.maculaOverride ?? this.detectedMacCenter;
  }
  get odDiameter(): number {
    return this.detectedOdDiameter;
  }

  // ── regions & features ──────────────────────────────────────────────
  showRegions = true;
  regionCircles: RegionCircle[] = [];
  regionSummaries: RegionSummary[] = [];
  polygonRegions = new Map<string, RegionName>();
  disabledPolygons = new Set<string>();
  featureVector: FeatureEntry[] = [];
  showFeatureVector = false;

  // ── classifier ──────────────────────────────────────────────────────
  prediction: Prediction | null = null;
  gradingScore: number | null = null;
  private modelLoaded = false;

  /** Only lesion classes (1–4) for the sidebar list. */
  get lesionClasses(): ClassPolygons[] {
    return this.segmentation.filter((c) =>
      (LESION_IDS as readonly number[]).includes(c.class_id),
    );
  }

  get lesionCount(): number {
    return this.lesionClasses.reduce((s, c) => s + c.polygons.length, 0);
  }

  get activeCount(): number {
    return this.lesionCount - this.disabledPolygons.size;
  }

  private wheelListener?: (e: WheelEvent) => void;

  constructor(
    public images: ImagesService,
    private ngZone: NgZone,
    public router: Router,
  ) {}

  // ══════════════════════════════════════════════════════════════════════
  //  LIFECYCLE
  // ══════════════════════════════════════════════════════════════════════

  ngOnInit(): void {
    const path = this.images.activeImage;
    if (!path) {
      this.router.navigate(['/gallery']);
      return;
    }
    this.imagePath = path;
    this.src = this.images.getCachedSrc(path);
    if (!this.src) {
      this.images.loadImageSrc(path).then((src) =>
        this.ngZone.run(() => {
          this.src = src;
        }),
      );
    }
  }

  ngAfterViewInit(): void {
    this.wheelListener = (e: WheelEvent) => {
      e.preventDefault();
      this.ngZone.run(() => this.onWheel(e));
    };
    this.viewportRef.nativeElement.addEventListener(
      'wheel',
      this.wheelListener,
      { passive: false },
    );
    if (this.src && this.imageRef?.nativeElement.complete) {
      this.onImageLoad();
    }
  }

  ngOnDestroy(): void {
    if (this.wheelListener) {
      this.viewportRef.nativeElement.removeEventListener(
        'wheel',
        this.wheelListener,
      );
    }
  }

  // ══════════════════════════════════════════════════════════════════════
  //  IMAGE LOAD
  // ══════════════════════════════════════════════════════════════════════

  async onImageLoad(): Promise<void> {
    const img = this.imageRef?.nativeElement;
    if (!img || !img.naturalWidth) return;
    this.imageNaturalWidth = img.naturalWidth;
    this.imageNaturalHeight = img.naturalHeight;
    this.fitToView();
    this.extract_roi();
  }

  async extract_roi(): Promise<void> {
    if (!this.imagePath) return;
    try {
      const json = await invoke<string>('extract_roi', {
        imagePath: this.imagePath,
      });
      this.ngZone.run(() => {
        this.ROI = JSON.parse(json)[0] as SegPolygon;
      });
    } catch (e) {
      console.error('Background extraction failed:', e);
    }
  }

  // ══════════════════════════════════════════════════════════════════════
  //  TRANSFORM HELPERS
  // ══════════════════════════════════════════════════════════════════════

  get cssTransform(): string {
    const { x, y, scale } = this.transform;
    return `translate3d(${x}px, ${y}px, 0px) scale(${scale})`;
  }

  get zoomPercent(): number {
    return Math.round(this.transform.scale * 100);
  }

  get roiClipPath(): string {
    if (!this.ROI || !this.imageNaturalWidth || !this.imageNaturalHeight)
      return 'none';
    const coords = this.ROI.points
      .map(
        ([x, y]) =>
          `${(x / this.imageNaturalWidth) * 100}% ${(y / this.imageNaturalHeight) * 100}%`,
      )
      .join(', ');
    return `polygon(${coords})`;
  }

  onOpacityChange(event: Event): void {
    this.overlayOpacity = parseFloat((event.target as HTMLInputElement).value);
  }

  onFovChange(event: Event): void {
    this.fov = parseInt((event.target as HTMLInputElement).value, 10);
  }

  // ══════════════════════════════════════════════════════════════════════
  //  SEGMENTATION
  // ══════════════════════════════════════════════════════════════════════

  async runSegmentation(): Promise<void> {
    if (!this.imagePath || this.isProcessing) return;
    this.isProcessing = true;
    try {
      // Load classifier model in parallel with segmentation
      const [json] = await Promise.all([
        invoke<string>('run_inference', {
          imagePath: this.imagePath,
          fov: this.fov,
        }),
        this.ensureModelLoaded(),
      ]);
      this.ngZone.run(() => {
        this.segmentation = JSON.parse(json);
        this.visibleClasses = new Set(
          this.segmentation
            .filter((c) =>
              (LESION_IDS as readonly number[]).includes(c.class_id),
            )
            .map((c) => c.class_id),
        );
        this.disabledPolygons.clear();
        this.maculaOverride = null;
        this.extractLandmarks();
        this.rebuildRegions();
      });
    } catch (e) {
      console.error('Inference failed:', e);
    } finally {
      this.ngZone.run(() => (this.isProcessing = false));
    }
  }

  clearSegmentation(): void {
    this.segmentation = [];
    this.visibleClasses.clear();
    this.disabledPolygons.clear();
    this.maculaOverride = null;
    this.regionCircles = [];
    this.regionSummaries = [];
    this.polygonRegions.clear();
    this.featureVector = [];
    this.showRegions = false;
    this.showFeatureVector = false;
    this.detectedOdCenter = null;
    this.detectedMacCenter = null;
    this.prediction = null;
  }

  toggleClass(classId: number): void {
    if (this.visibleClasses.has(classId)) this.visibleClasses.delete(classId);
    else this.visibleClasses.add(classId);
    this.rebuildFeatureVector();
  }

  isClassVisible(classId: number): boolean {
    return this.visibleClasses.has(classId);
  }

  polyPoints(points: [number, number][]): string {
    return points.map(([x, y]) => `${x},${y}`).join(' ');
  }

  classStyle(classId: number) {
    return (
      CLASS_STYLES[classId] ?? {
        fill: 'rgba(255,255,255,0.25)',
        stroke: 'white',
      }
    );
  }

  // ══════════════════════════════════════════════════════════════════════
  //  LANDMARKS & REGIONS
  // ══════════════════════════════════════════════════════════════════════

  private extractLandmarks(): void {
    const odClass = this.segmentation.find((c) => c.class_id === OD_CLASS);
    const macClass = this.segmentation.find((c) => c.class_id === MACULA_CLASS);

    if (odClass?.polygons.length) {
      const allPts = odClass.polygons.flatMap((p) => p.points);
      this.detectedOdCenter = polygonCentroid(allPts);
      const bbox = polygonBBox(allPts);
      this.detectedOdDiameter = Math.max(
        bbox.maxX - bbox.minX,
        bbox.maxY - bbox.minY,
      );
    } else {
      this.detectedOdCenter = null;
      this.detectedOdDiameter = 150;
    }

    if (macClass?.polygons.length) {
      const allPts = macClass.polygons.flatMap((p) => p.points);
      this.detectedMacCenter = polygonCentroid(allPts);
    } else {
      this.detectedMacCenter = null;
    }
  }

  /** Recompute region circles, polygon→region assignments, summaries, and feature vector. */
  rebuildRegions(): void {
    this.regionCircles = computeRegionCircles(
      this.odCenter,
      this.macCenter,
      this.odDiameter,
    );
    this.polygonRegions = assignPolygonRegions(
      this.segmentation,
      this.odCenter,
      this.macCenter,
      this.odDiameter,
    );
    this.rebuildFeatureVector();
  }

  /** Recompute only the feature vector and region summaries (after polygon toggle). */
  rebuildFeatureVector(): void {
    this.featureVector = computeFeatureVector(
      this.segmentation,
      this.polygonRegions,
      this.disabledPolygons,
      this.visibleClasses,
    );
    this.regionSummaries = computeRegionSummaries(
      this.polygonRegions,
      this.disabledPolygons,
    );
    this.runPrediction();
  }

  // ── classifier ──────────────────────────────────────────────────────

  /** Load model weights (idempotent — safe to call multiple times). */
  async ensureModelLoaded(): Promise<void> {
    if (this.modelLoaded || (getModel() && getScoreModel())) {
      this.modelLoaded = true;
      return;
    }
    try {
      // Path relative to your Tauri public/assets folder
      await loadModel('/assets/dr_screening_pipeline.json');
      await loadScoreModel('/assets/dr_grading_pipeline.json');

      this.modelLoaded = true;
    } catch (e) {
      console.warn('Classifier model not available:', e);
    }
  }

  private runPrediction(): void {
    if (!this.modelLoaded || !this.featureVector.length) {
      this.prediction = null;
      return;
    }
    try {
      this.prediction = predict(this.featureVector);
      this.gradingScore = predictScore(this.featureVector);
    } catch (e) {
      console.warn('Prediction failed:', e);
      this.prediction = null;
      this.gradingScore = null;
    }
  }

  // ══════════════════════════════════════════════════════════════════════
  //  POLYGON TOGGLING (shutdown / reactivate)
  // ══════════════════════════════════════════════════════════════════════

  isPolygonDisabled(classId: number, polyIndex: number): boolean {
    return this.disabledPolygons.has(polyKey(classId, polyIndex));
  }

  /** Called when clicking an individual polygon on the canvas. */
  onPolygonClick(event: Event, classId: number, polyIndex: number): void {
    if (this.panMoved) return; // was a pan, not a click
    event.stopPropagation();
    const key = polyKey(classId, polyIndex);
    if (this.disabledPolygons.has(key)) this.disabledPolygons.delete(key);
    else this.disabledPolygons.add(key);
    this.rebuildFeatureVector();
  }

  /** Toggle all lesion polygons in a given region. */
  toggleRegionLesions(region: RegionName): void {
    const keys = [...this.polygonRegions.entries()]
      .filter(([, r]) => r === region)
      .map(([k]) => k);
    if (!keys.length) return;

    // If all are active → disable all. Otherwise → enable all.
    const allActive = keys.every((k) => !this.disabledPolygons.has(k));
    for (const k of keys) {
      if (allActive) this.disabledPolygons.add(k);
      else this.disabledPolygons.delete(k);
    }
    this.rebuildFeatureVector();
  }

  /** Are all lesions in a region currently active? */
  isRegionAllActive(region: RegionName): boolean {
    const summary = this.regionSummaries.find((s) => s.region === region);
    return !!summary && summary.total > 0 && summary.active === summary.total;
  }

  toggleRegions(): void {
    this.showRegions = !this.showRegions;
  }
  toggleFeatureVector(): void {
    this.showFeatureVector = !this.showFeatureVector;
  }

  // ══════════════════════════════════════════════════════════════════════
  //  MACULA DRAG
  // ══════════════════════════════════════════════════════════════════════

  onMaculaDragStart(event: MouseEvent): void {
    event.stopPropagation(); // prevent viewport pan
    event.preventDefault();
    this.isDraggingMacula = true;
  }

  /** Convert a mouse event to image-space coordinates. */
  private mouseToImage(e: MouseEvent): [number, number] {
    const rect = this.viewportRef.nativeElement.getBoundingClientRect();
    const imgX =
      (e.clientX - rect.left - this.transform.x) / this.transform.scale;
    const imgY =
      (e.clientY - rect.top - this.transform.y) / this.transform.scale;
    return [imgX, imgY];
  }

  // ══════════════════════════════════════════════════════════════════════
  //  ZOOM & PAN
  // ══════════════════════════════════════════════════════════════════════

  fitToView(): void {
    const vp = this.viewportRef?.nativeElement;
    if (!vp || !this.imageNaturalWidth || !this.imageNaturalHeight) return;
    const width = vp.clientWidth;
    const height = vp.clientHeight;
    if (width === 0 || height === 0) {
      setTimeout(() => this.fitToView(), 10);
      return;
    }
    const scaleX = width / this.imageNaturalWidth;
    const scaleY = height / this.imageNaturalHeight;
    const scale = Math.min(scaleX, scaleY) * 0.92;
    const x = (width - this.imageNaturalWidth * scale) / 2;
    const y = (height - this.imageNaturalHeight * scale) / 2;
    this.transform = { x, y, scale };
  }

  resetZoom(): void {
    const vp = this.viewportRef.nativeElement;
    const x = (vp.clientWidth - this.imageNaturalWidth) / 2;
    const y = (vp.clientHeight - this.imageNaturalHeight) / 2;
    this.transform = { x, y, scale: 1 };
  }

  private onWheel(e: WheelEvent): void {
    const vp = this.viewportRef.nativeElement;
    const rect = vp.getBoundingClientRect();
    const cursorX = e.clientX - rect.left;
    const cursorY = e.clientY - rect.top;
    const delta = -e.deltaY * ZOOM_SENSITIVITY;
    const factor = Math.exp(delta);
    const newScale = Math.min(
      MAX_SCALE,
      Math.max(MIN_SCALE, this.transform.scale * factor),
    );
    const ratio = newScale / this.transform.scale;
    const newX = cursorX - (cursorX - this.transform.x) * ratio;
    const newY = cursorY - (cursorY - this.transform.y) * ratio;
    this.transform = { x: newX, y: newY, scale: newScale };
  }

  onMouseDown(e: MouseEvent): void {
    if (e.button !== 0) return;
    e.preventDefault();
    this.isPanning = true;
    this.panMoved = false;
    this.panStart = { x: e.clientX, y: e.clientY };
    this.transformAtPanStart = { ...this.transform };
  }

  onMouseMove(e: MouseEvent): void {
    // Macula drag takes priority
    if (this.isDraggingMacula) {
      this.maculaOverride = this.mouseToImage(e);
      // Live-update circles (lightweight; feature vector rebuilt on drop)
      this.regionCircles = computeRegionCircles(
        this.odCenter,
        this.macCenter,
        this.odDiameter,
      );
      return;
    }

    if (!this.isPanning) return;
    const dx = e.clientX - this.panStart.x;
    const dy = e.clientY - this.panStart.y;
    if (Math.abs(dx) > CLICK_THRESHOLD || Math.abs(dy) > CLICK_THRESHOLD) {
      this.panMoved = true;
    }
    this.transform = {
      ...this.transform,
      x: this.transformAtPanStart.x + dx,
      y: this.transformAtPanStart.y + dy,
    };
  }

  onMouseUp(): void {
    if (this.isDraggingMacula) {
      this.isDraggingMacula = false;
      this.rebuildRegions(); // full recompute: assignments + feature vector
      return;
    }
    this.isPanning = false;
  }

  countPolygons(acc: number, cls: ClassPolygons): number {
    return acc + cls.polygons.length;
  }
}
