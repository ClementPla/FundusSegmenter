use crate::utils::image::{ClassPolygons, SerializedPolygon};

// ── OD / Macula post-processing ───────────────────────────────────────────────

/// Centroid of a polygon's point cloud.
fn centroid(points: &[[f32; 2]]) -> [f32; 2] {
    let n = points.len() as f32;
    let sx: f32 = points.iter().map(|p| p[0]).sum();
    let sy: f32 = points.iter().map(|p| p[1]).sum();
    [sx / n, sy / n]
}

/// Axis-aligned bounding-box area — fast proxy for polygon size.
fn bbox_area(points: &[[f32; 2]]) -> f32 {
    let min_x = points.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
    let max_x = points.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
    let min_y = points.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
    let max_y = points.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
    (max_x - min_x) * (max_y - min_y)
}

/// Bounding-box diagonal — used as a size reference for alignment tolerance.
fn bbox_diagonal(points: &[[f32; 2]]) -> f32 {
    let min_x = points.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
    let max_x = points.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
    let min_y = points.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
    let max_y = points.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
    let dw = max_x - min_x;
    let dh = max_y - min_y;
    (dw * dw + dh * dh).sqrt()
}

/// Keep only the largest polygon, discarding any candidate whose bbox area is
/// less than `1/area_ratio` of the largest. This removes small confounders
/// (vessels, bright spots) that are anatomically implausible OD duplicates.
fn keep_dominant(polygons: Vec<SerializedPolygon>, area_ratio: f32) -> Vec<SerializedPolygon> {
    if polygons.is_empty() {
        return polygons;
    }

    // Find the dominant (largest) polygon
    let dominant_idx = polygons
        .iter()
        .enumerate()
        .map(|(i, p)| (i, bbox_area(&p.points)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let dominant_area = bbox_area(&polygons[dominant_idx].points);
    let threshold = dominant_area / area_ratio;

    polygons
        .into_iter()
        .enumerate()
        .filter(|(i, p)| *i == dominant_idx || bbox_area(&p.points) >= threshold)
        .map(|(_, p)| p)
        // After threshold filter, keep only the single dominant
        .take(1)
        .collect()
}

fn synthetic_circle(cx: f32, cy: f32, radius: f32, n_pts: usize) -> SerializedPolygon {
    let pts = (0..n_pts)
        .map(|i| {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / n_pts as f32;
            [cx + radius * angle.cos(), cy + radius * angle.sin()]
        })
        .collect();
    SerializedPolygon { points: pts }
}

/// Post-process the od_mac label map polygons:
///
/// - OD  (class_id == od_class):  keep only one; discard confounders smaller
///   than 1/10 of the dominant candidate's bounding-box area.
///
/// - Mac (class_id == mac_class): keep only one; it must be within
///   `alignment_tolerance` (expressed as a fraction of the OD diagonal) of
///   the OD centroid's **y-coordinate** (horizontal alignment on a fundus
///   image). If no macula candidate satisfies the constraint, keep the largest
///   remaining one with a warning.
///
/// Returns the cleaned list of ClassPolygons (empty classes are dropped).
pub fn postprocess_od_mac(
    classes: Vec<ClassPolygons>,
    od_class: usize,
    mac_class: usize,
    alignment_tolerance: f32,
    image_width: u32,
    image_height: u32,
) -> Vec<ClassPolygons> {
    let mut od_polys: Option<Vec<SerializedPolygon>> = None;
    let mut mac_polys: Option<Vec<SerializedPolygon>> = None;
    let mut rest: Vec<ClassPolygons> = Vec::new();

    for cls in classes {
        if cls.class_id == od_class {
            od_polys = Some(cls.polygons);
        } else if cls.class_id == mac_class {
            mac_polys = Some(cls.polygons);
        } else {
            rest.push(cls);
        }
    }

    // ── 1. Optic disc ─────────────────────────────────────────────────────────
    let od_polys = od_polys.map(|p| keep_dominant(p, 10.0)).unwrap_or_default();

    let od_centroid: Option<[f32; 2]> = od_polys.first().map(|p| centroid(&p.points));
    let od_diagonal: f32 = od_polys
        .first()
        .map(|p| bbox_diagonal(&p.points))
        .unwrap_or(image_width.min(image_height) as f32 * 0.15); // sane default if OD missing
    let vertical_slack = od_diagonal * alignment_tolerance;

    // ── 2. Macula ─────────────────────────────────────────────────────────────
    let mac_polys = {
        let candidates = mac_polys
            .map(|p| keep_dominant(p, 10.0))
            .unwrap_or_default();

        if candidates.is_empty() {
            // No macula detected — synthesize one.
            // Anatomy: macula is ~2.5 OD diameters temporal to OD, same y.
            // If OD is also absent, fall back to image center.
            let (cx, cy) = match od_centroid {
                Some(od_c) => {
                    // Determine which side is temporal by checking if OD is
                    // left or right of image center (nasal vs temporal).
                    let img_cx = image_width as f32 / 2.0;
                    let temporal_offset = od_diagonal * 2.5;
                    let cx = if od_c[0] < img_cx {
                        // OD is on the left → right eye → macula is to the right
                        od_c[0] + temporal_offset
                    } else {
                        // OD is on the right → left eye → macula is to the left
                        od_c[0] - temporal_offset
                    };
                    (cx.clamp(0.0, image_width as f32), od_c[1])
                }
                None => {
                    println!("[postprocess] no OD found; placing macula at image center");
                    (image_width as f32 / 2.0, image_height as f32 / 2.0)
                }
            };

            // Radius: roughly 0.6 × OD radius is a typical macula foveal zone
            let radius = (od_diagonal / 2.0) * 0.6;
            println!(
                "[postprocess] macula not found — synthesizing at ({cx:.1}, {cy:.1}) r={radius:.1}px"
            );
            vec![synthetic_circle(cx, cy, radius, 32)]
        } else {
            // Alignment check (same as before)
            match od_centroid {
                None => {
                    println!("[postprocess] no OD found; skipping macula alignment check");
                    candidates
                }
                Some(od_c) => {
                    let best = candidates
                        .iter()
                        .enumerate()
                        .map(|(i, p)| {
                            let c = centroid(&p.points);
                            let dy = (c[1] - od_c[1]).abs();
                            (i, c, dy)
                        })
                        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

                    if let Some((idx, mac_c, dy)) = best {
                        if dy <= vertical_slack {
                            println!(
                                "[postprocess] macula accepted: Δy={:.1}px (slack={:.1}px)",
                                dy, vertical_slack
                            );
                        } else {
                            println!(
                                "[postprocess] macula misaligned: Δy={:.1}px exceeds slack={:.1}px \
                                 (OD_y={:.1}, mac_y={:.1}) — keeping as fallback",
                                dy, vertical_slack, od_c[1], mac_c[1]
                            );
                        }
                        vec![candidates.into_iter().nth(idx).unwrap()]
                    } else {
                        candidates
                    }
                }
            }
        }
    };

    // ── Reassemble ────────────────────────────────────────────────────────────
    let mut result = rest;
    if !od_polys.is_empty() {
        result.push(ClassPolygons {
            class_id: od_class,
            polygons: od_polys,
        });
    }
    if !mac_polys.is_empty() {
        result.push(ClassPolygons {
            class_id: mac_class,
            polygons: mac_polys,
        });
    }
    result.sort_by_key(|c| c.class_id);
    result
}
