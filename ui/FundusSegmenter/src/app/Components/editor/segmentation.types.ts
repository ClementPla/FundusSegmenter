export interface SegPolygon {
  points: [number, number][];
}

export interface ClassPolygons {
  class_id: number;
  polygons: SegPolygon[];
}

// Must match Rust's LABEL_COLORS order
export const CLASS_STYLES: Record<number, { fill: string; stroke: string }> = {
  1: { fill: 'rgba(75,  138, 159, 0.35)', stroke: 'rgb(75,  138, 159)' },
  2: { fill: 'rgba(20,  89,  126, 0.35)', stroke: 'rgb(20,  89,  126)' },
  3: { fill: 'rgba(91,  41,  148, 0.35)', stroke: 'rgb(91,  41,  148)' },
  4: { fill: 'rgba(147, 61,  147, 0.35)', stroke: 'rgb(147, 61,  147)' },
  5: { fill: 'rgb(0, 255, 13)', stroke: 'rgb(147, 61,  147)' },
  6: { fill: 'rgb(27, 2, 253)', stroke: 'rgb(147, 61,  147)' },
};

export const CLASS_NAMES: Record<number, string> = {
  1: 'Cotton Wool Spots',
  2: 'Hard Exudates',
  3: 'Hemorrhages',
  4: 'Microaneurysms',
  5: 'Optic Disk',
  6: 'Macula',
};
