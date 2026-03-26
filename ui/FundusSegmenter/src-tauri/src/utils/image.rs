use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::contours::{find_contours, BorderType};

// ── Polygon types ─────────────────────────────────────────────────────────────
const SIMPLIFICATION_EPSILON: f64 = 1.5; // px — Douglas-Peucker tolerance
const MIN_CONTOUR_AREA: f64 = 150.0; // px² at original resolution — filters noise
const BLACK_THRESHOLD: u8 = 10;

#[derive(serde::Serialize)]
pub struct SerializedPolygon {
    pub points: Vec<[f32; 2]>,
}

#[derive(serde::Serialize)]
pub struct ClassPolygons {
    pub class_id: usize,
    pub polygons: Vec<SerializedPolygon>,
}

// ── Geometry helpers ──────────────────────────────────────────────────────────

fn polygon_area(pts: &[[f64; 2]]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let area: f64 = (0..n)
        .map(|i| {
            let j = (i + 1) % n;
            pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
        })
        .sum();
    (area / 2.0).abs()
}

fn perp_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len == 0.0 {
        return ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt();
    }
    ((dy * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]) / len).abs()
}

fn douglas_peucker(pts: &[[f64; 2]], eps: f64) -> Vec<[f64; 2]> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let end = *pts.last().unwrap();
    let (max_i, max_d) = pts[1..pts.len() - 1]
        .iter()
        .enumerate()
        .map(|(i, &p)| (i + 1, perp_dist(p, pts[0], end)))
        .fold(
            (0, 0.0f64),
            |(mi, md), (i, d)| if d > md { (i, d) } else { (mi, md) },
        );

    if max_d > eps {
        let mut r = douglas_peucker(&pts[..=max_i], eps);
        r.pop();
        r.extend(douglas_peucker(&pts[max_i..], eps));
        r
    } else {
        vec![pts[0], end]
    }
}

// ── Contour extraction ────────────────────────────────────────────────────────

pub fn extract_class_polygons(label_map: &GrayImage, class_id: u8) -> Vec<SerializedPolygon> {
    extract_class_polygons_scaled(label_map, class_id, label_map.width(), label_map.width(), 0, 0, 0, 0)
}


pub fn extract_class_polygons_scaled(
    label_map: &GrayImage,             // 1024×1024 model output
    class_id: u8,
    model_size: u32,                   // MODEL_INPUT_SIZE (1024)
    square_side: u32,                  // size after pad_to_square
    pad_x: u32, pad_y: u32,            // padding offsets
    crop_x: u32, crop_y: u32,          // crop offset in original image
) -> Vec<SerializedPolygon> {
    let scale = square_side as f64 / model_size as f64;

    let binary: GrayImage = ImageBuffer::from_fn(
        label_map.width(), label_map.height(),
        |x, y| if label_map.get_pixel(x, y).0[0] == class_id { Luma([255u8]) } else { Luma([0u8]) },
    );

    find_contours::<i32>(&binary)
        .into_iter()
        .filter(|c| c.border_type == BorderType::Outer && c.points.len() >= 6)
        .filter_map(|c| {
            // Transform each point: model → square → unpad → uncrop
            let pts: Vec<[f64; 2]> = c.points.iter().map(|p| {
                let sx = p.x as f64 * scale;
                let sy = p.y as f64 * scale;
                let ux = sx - pad_x as f64 + crop_x as f64;
                let uy = sy - pad_y as f64 + crop_y as f64;
                [ux, uy]
            }).collect();

            if polygon_area(&pts) < MIN_CONTOUR_AREA { return None; }
            let simplified = douglas_peucker(&pts, SIMPLIFICATION_EPSILON * scale);
            if simplified.len() < 3 { return None; }
            Some(SerializedPolygon {
                points: simplified.iter().map(|p| [p[0] as f32, p[1] as f32]).collect(),
            })
        })
        .collect()
}


// ── Image helpers ─────────────────────────────────────────────────────────────

pub struct CropBounds {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}
pub fn find_content_bounds(img: &RgbImage) -> CropBounds {
    let (width, height) = img.dimensions();
    let is_black = |p: &Rgb<u8>| p.0.iter().all(|&c| c <= BLACK_THRESHOLD);

    let top = (0..height)
        .find(|&y| (0..width).any(|x| !is_black(img.get_pixel(x, y))))
        .unwrap_or(0);
    let bottom = (0..height)
        .rev()
        .find(|&y| (0..width).any(|x| !is_black(img.get_pixel(x, y))))
        .unwrap_or(height - 1);
    let left = (0..width)
        .find(|&x| (top..=bottom).any(|y| !is_black(img.get_pixel(x, y))))
        .unwrap_or(0);
    let right = (0..width)
        .rev()
        .find(|&x| (top..=bottom).any(|y| !is_black(img.get_pixel(x, y))))
        .unwrap_or(width - 1);

    let w = right - left + 1;
    let h = bottom - top + 1;

    // Sanity check: if the detected content covers less than 50% of the
    // image area, the bounds are likely wrong (e.g. bright-background image
    // with no black border).  Fall back to the full image.
    let content_area = w as u64 * h as u64;
    let image_area = width as u64 * height as u64;

    if content_area * 2 < image_area {
        CropBounds { x: 0, y: 0, w: width, h: height }
    } else {
        CropBounds { x: left, y: top, w, h }
    }
}

pub fn pad_to_square_rgb(img: &RgbImage) -> (RgbImage, u32, u32) {
    let (w, h) = img.dimensions();
    let side = w.max(h);
    let offset_x = (side - w) / 2;
    let offset_y = (side - h) / 2;
    let mut canvas = ImageBuffer::from_pixel(side, side, Rgb([0u8, 0, 0]));
    image::imageops::replace(&mut canvas, img, offset_x as i64, offset_y as i64);
    (canvas, offset_x, offset_y)
}


#[tauri::command]
pub async fn extract_roi(image_path: String) -> Result<String, String> {
    let original = image::open(&image_path)
        .map_err(|e| format!("failed to open image: {e}"))?
        .to_rgb8();

    let bounds = find_content_bounds(&original);

    // Build a single polygon from the bounding rectangle (4 corners, clockwise)
    let polygon = vec![SerializedPolygon {
        points: vec![
            [bounds.x as f32,              bounds.y as f32],
            [bounds.x as f32 + bounds.w as f32, bounds.y as f32],
            [bounds.x as f32 + bounds.w as f32, bounds.y as f32 + bounds.h as f32],
            [bounds.x as f32,              bounds.y as f32 + bounds.h as f32],
        ],
    }];

    let json = serde_json::to_string(&polygon)
        .map_err(|e| format!("JSON serialization failed: {e}"))?;
    Ok(json)
}
