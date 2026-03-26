/**
 * 48-component feature vector computation.
 *
 * Layout: 3 metrics × 4 regions × 4 lesion types = 48
 *   metrics:  count, total_area, mean_area
 *   regions:  OD region, macula region, 1OD-2OD from macula, elsewhere
 *   lesions:  COTTON_WOOL_SPOT, EXUDATES, HEMORRHAGES, MICROANEURYSMS
 */

import { ClassPolygons } from '../segmentation.types';
import { polygonArea, polygonCentroid } from './geometry';
import { classifyPoint, REGION_ORDER, type RegionName } from './regions';

// ── constants ───────────────────────────────────────────────────────────

export const LESION_IDS = [1, 2, 3, 4] as const;
export type LesionId = (typeof LESION_IDS)[number];

export const LESION_NAMES: Record<number, string> = {
  1: 'COTTON_WOOL_SPOT',
  2: 'EXUDATES',
  3: 'HEMORRHAGES',
  4: 'MICROANEURYSMS',
};

export const METRICS = ['count', 'total_area', 'mean_area'] as const;

export interface FeatureEntry {
  metric: string;
  region: RegionName;
  lesion: string;
  value: number;
}

// ── polygon key helper ──────────────────────────────────────────────────

export function polyKey(classId: number, polyIndex: number): string {
  return `${classId}-${polyIndex}`;
}

// ── region assignment (per polygon) ─────────────────────────────────────

/**
 * Assign every lesion polygon to a spatial region.
 * Returns a Map keyed by `"classId-polyIndex"`.
 */
export function assignPolygonRegions(
  segmentation: ClassPolygons[],
  odCenter: [number, number] | null,
  macCenter: [number, number] | null,
  odDiameter: number,
): Map<string, RegionName> {
  const map = new Map<string, RegionName>();

  for (const cls of segmentation) {
    if (!(LESION_IDS as readonly number[]).includes(cls.class_id)) continue;
    for (let i = 0; i < cls.polygons.length; i++) {
      const centroid = polygonCentroid(cls.polygons[i].points);
      const region = classifyPoint(centroid, odCenter, macCenter, odDiameter);
      map.set(polyKey(cls.class_id, i), region);
    }
  }
  return map;
}

// ── feature vector ──────────────────────────────────────────────────────

/**
 * Compute the 48-entry feature vector from the current segmentation state.
 * Disabled polygons are excluded from aggregation.
 */
export function computeFeatureVector(
  segmentation: ClassPolygons[],
  polygonRegions: Map<string, RegionName>,
  disabledPolygons: Set<string>,
  visibleClasses: Set<number>,
): FeatureEntry[] {
  // Accumulator: region|lesionId → { count, totalArea }
  const acc = new Map<string, { count: number; totalArea: number }>();
  for (const region of REGION_ORDER) {
    for (const lid of LESION_IDS) {
      acc.set(`${region}|${lid}`, { count: 0, totalArea: 0 });
    }
  }

  for (const cls of segmentation) {
    if (!(LESION_IDS as readonly number[]).includes(cls.class_id)) continue;
    for (let i = 0; i < cls.polygons.length; i++) {
      if (!visibleClasses.has(cls.class_id)) continue;
      const key = polyKey(cls.class_id, i);
      if (disabledPolygons.has(key)) continue;

      const region = polygonRegions.get(key);
      if (!region) continue;

      const area = polygonArea(cls.polygons[i].points);
      const entry = acc.get(`${region}|${cls.class_id}`)!;
      entry.count++;
      entry.totalArea += area;
    }
  }

  // Flatten into 48 entries
  const features: FeatureEntry[] = [];
  for (const metric of METRICS) {
    for (const region of REGION_ORDER) {
      for (const lid of LESION_IDS) {
        const { count, totalArea } = acc.get(`${region}|${lid}`)!;
        let value: number;
        if (metric === 'count') value = count;
        else if (metric === 'total_area') value = totalArea;
        else value = count > 0 ? totalArea / count : 0;
        features.push({ metric, region, lesion: LESION_NAMES[lid], value });
      }
    }
  }
  return features;
}
