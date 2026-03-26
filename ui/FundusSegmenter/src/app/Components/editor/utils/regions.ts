/**
 * Spatial region definitions and classification logic.
 *
 * Regions (distances from lesion centroid):
 *   - OD region:            ≤ 0.5 × OD diameter from OD centre
 *   - Macula region:        ≤ 1   × OD diameter from macula centre
 *   - 1OD-2OD from macula:  ≤ 2   × OD diameter from macula centre
 *   - Elsewhere:            everything else
 *
 * Priority: OD > Macula > Perimacular > Elsewhere
 */

import { distance } from './geometry';

// ── constants ───────────────────────────────────────────────────────────

export const REGION_ORDER = [
  'OD region',
  'macula region',
  '1OD-2OD from macula',
  'elsewhere',
] as const;

export type RegionName = (typeof REGION_ORDER)[number];

export interface RegionCircle {
  cx: number;
  cy: number;
  r: number;
  color: string;
  dash: string;
  label: string;
  region: RegionName;
}

export interface RegionSummary {
  region: RegionName;
  active: number;
  total: number;
}

const REGION_DISPLAY: Record<
  string,
  { color: string; dash: string; label: string }
> = {
  'OD region': { color: 'lime', dash: '', label: 'OD region (0.5 OD)' },
  'macula region': { color: 'cyan', dash: '', label: 'Macula region (1 OD)' },
  '1OD-2OD from macula': {
    color: 'cyan',
    dash: '16 8',
    label: 'Perimacular (2 OD)',
  },
};

// ── classification ──────────────────────────────────────────────────────

export function classifyPoint(
  point: [number, number],
  odCenter: [number, number] | null,
  macCenter: [number, number] | null,
  odDiameter: number,
): RegionName {
  const odRadius = odDiameter / 2;
  const dOd = odCenter ? distance(point, odCenter) : Infinity;
  const dMac = macCenter ? distance(point, macCenter) : Infinity;

  if (dOd <= odRadius) return 'OD region';
  if (dMac <= odDiameter) return 'macula region';
  if (dMac <= 2 * odDiameter) return '1OD-2OD from macula';
  return 'elsewhere';
}

// ── circle generation ───────────────────────────────────────────────────

export function computeRegionCircles(
  odCenter: [number, number] | null,
  macCenter: [number, number] | null,
  odDiameter: number,
): RegionCircle[] {
  const circles: RegionCircle[] = [];
  const odRadius = odDiameter / 2;

  if (odCenter) {
    circles.push({
      cx: odCenter[0],
      cy: odCenter[1],
      r: odRadius,
      ...REGION_DISPLAY['OD region'],
      region: 'OD region',
    });
  }
  if (macCenter) {
    circles.push({
      cx: macCenter[0],
      cy: macCenter[1],
      r: odDiameter,
      ...REGION_DISPLAY['macula region'],
      region: 'macula region',
    });
    circles.push({
      cx: macCenter[0],
      cy: macCenter[1],
      r: 2 * odDiameter,
      ...REGION_DISPLAY['1OD-2OD from macula'],
      region: '1OD-2OD from macula',
    });
  }
  return circles;
}

// ── region summaries ────────────────────────────────────────────────────

export function computeRegionSummaries(
  polygonRegions: Map<string, RegionName>,
  disabledPolygons: Set<string>,
): RegionSummary[] {
  const totals = new Map<RegionName, { active: number; total: number }>();
  for (const region of REGION_ORDER) {
    totals.set(region, { active: 0, total: 0 });
  }

  for (const [key, region] of polygonRegions) {
    const entry = totals.get(region)!;
    entry.total++;
    if (!disabledPolygons.has(key)) entry.active++;
  }

  return REGION_ORDER.map((region) => ({
    region,
    ...totals.get(region)!,
  }));
}
