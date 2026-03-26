use image::{GrayImage, ImageBuffer, Luma};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use tauri::Manager;
use crate::utils::image::{
  extract_class_polygons_scaled,
  pad_to_square_rgb,
  ClassPolygons,
  find_content_bounds,
};
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use ort::session::RunOptions;
use crate::dl::postprocess::postprocess_od_mac;
use crate::dl::model_path::model_path;


// ── Fixed model input size ────────────────────────────────────────────────────
// The ONNX model was exported with static axes at 1024×1024.
// FoV adjustment is handled as a preprocessing scale factor, not by
// changing the model input dimensions.
const MODEL_INPUT_SIZE: u32 = 1024;
const BASE_FOV: f32 = 45.0;

// ── Session ───────────────────────────────────────────────────────────────────

fn get_session(model_path: &str) -> &'static Mutex<Session> {
    static SESSION: OnceLock<Mutex<Session>> = OnceLock::new();
    SESSION.get_or_init(|| {
        let t = Instant::now();
        println!("[inference] registering execution providers: CUDA → CPU fallback");

        let session = Session::builder()
            .expect("failed to create session builder")
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .expect("failed to set optimization level")
            .with_intra_threads(4)
            .expect("failed to set intra threads")
            .with_config_entry("session.gpu_allocator_config", "arena_extend_strategy:kSameAsRequested")
            .expect("failed to set arena strategy")
            .with_execution_providers([
                CUDAExecutionProvider::default()
                    .with_conv_max_workspace(true)
                    .build(),
                CPUExecutionProvider::default().build(),
            ])
            .expect("failed to register execution providers")
            .commit_from_file(model_path)
            .expect("failed to load ONNX model");

        println!("[inference] model loaded in {:.2?}", t.elapsed());
        Mutex::new(session)
    })
}


fn tensor_to_label_map(data: &[i64], shape: &[i64]) -> GrayImage {
    let h = shape[1] as u32;
    let w = shape[2] as u32;
    ImageBuffer::from_fn(w, h, |x, y| {
        let idx = y as usize * w as usize + x as usize;
        Luma([data[idx].clamp(0, 255) as u8])
    })
}

#[tauri::command]
pub async fn run_inference(handle: tauri::AppHandle, image_path: String, fov: f32) -> Result<String, String> {
    // ── Resolve model from AppLocalData (downloaded on first launch) ─────
    let model_buf = model_path(&handle)?;
    if !model_buf.exists() {
        return Err("Model not found. Please download it first from the setup dialog.".into());
    }
    let model_path_str = model_buf.to_str().ok_or("Model path contains invalid UTF-8")?;

    // ── FoV → preprocessing scale factor ────────────────────────────────
    let _scale_factor: f32 = BASE_FOV / fov.clamp(10.0, 200.0);

    let t_total = Instant::now();

    // 1. Load
    let t = Instant::now();
    let original = image::open(&image_path)
        .map_err(|e| format!("failed to open image: {e}"))?
        .to_rgb8();
    let (orig_w, orig_h) = original.dimensions();
    println!(
        "[inference] 1. load            {:.2?}  ({orig_w}×{orig_h})",
        t.elapsed()
    );

    // 2. Crop
    let t = Instant::now();
    let bounds = find_content_bounds(&original);
    let cropped =
        image::imageops::crop_imm(&original, bounds.x, bounds.y, bounds.w, bounds.h).to_image();
    println!(
        "[inference] 2. crop            {:.2?}  → {}×{} @ ({},{})",
        t.elapsed(),
        bounds.w,
        bounds.h,
        bounds.x,
        bounds.y
    );

    // 3. Pad
    let t = Instant::now();
    let (squared, pad_x, pad_y) = pad_to_square_rgb(&cropped);
    let square_side = squared.dimensions().0;
    println!(
        "[inference] 3. pad             {:.2?}  → {square_side}×{square_side}",
        t.elapsed()
    );

    // 4. Resize to fixed model input size
    let t = Instant::now();
    let resized = image::imageops::resize(
        &squared,
        (MODEL_INPUT_SIZE as f32 * _scale_factor) as u32,
        (MODEL_INPUT_SIZE as f32 * _scale_factor) as u32,
        image::imageops::FilterType::Triangle,
    );
    println!(
        "[inference] 4. resize→model    {:.2?}  → {MODEL_INPUT_SIZE}×{MODEL_INPUT_SIZE}  (fov={fov}°)",
        t.elapsed()
    );

    // 5. Build tensor [1, 3, H, W]
    let t = Instant::now();
    let (w, h) = resized.dimensions();
    let tensor_data: Vec<f32> = {
        let pixels = resized.as_raw();
        let n = (h * w) as usize;
        let mut v = vec![0f32; 3 * n];
        for i in 0..n {
            v[i] = pixels[i * 3] as f32 / 255.0;
            v[n + i] = pixels[i * 3 + 1] as f32 / 255.0;
            v[2 * n + i] = pixels[i * 3 + 2] as f32 / 255.0;
        }
        v
    };
    let input_tensor =
        Tensor::<f32>::from_array((vec![1usize, 3, h as usize, w as usize], tensor_data))
            .map_err(|e| format!("failed to create input tensor: {e}"))?;
    println!("[inference] 5. to tensor       {:.2?}", t.elapsed());

    // 6. Inference via IO binding
    let t = Instant::now();
    let session = get_session(model_path_str);
    let mut session = session
        .lock()
        .map_err(|_| "failed to acquire session lock")?;
    let mut run_options = RunOptions::new().map_err(|e| e.to_string())?;
    run_options.add_config_entry("memory.enable_shrinkage", "gpu:0").map_err(|e| e.to_string())?;
    let mut binding = session
        .create_binding()
        .map_err(|e| format!("failed to create binding: {e}"))?;

    binding
        .bind_input("input", &input_tensor)
        .map_err(|e| format!("failed to bind input: {e}"))?;

    binding
        .bind_output_to_device("lesions", &session.allocator().memory_info())
        .map_err(|e| format!("failed to bind output: {e}"))?;

    binding
        .bind_output_to_device("od_mac", &session.allocator().memory_info())
        .map_err(|e| format!("failed to bind output: {e}"))?;

    let mut outputs = session
        .run_binding_with_options(&binding, &run_options)
        .map_err(|e| format!("inference failed: {e}"))?;
    println!("[inference] 6. inference       {:.2?}", t.elapsed());

    // 7. Extract
    let t = Instant::now();
    let lesion_output = outputs.remove("lesions").ok_or("lesions tensor missing")?;
    let od_mac_output = outputs.remove("od_mac").ok_or("od_mac tensor missing")?;
    let (shape, raw_data) = lesion_output
        .try_extract_tensor::<i64>()
        .map_err(|e| format!("failed to extract output: {e}"))?;

    let (_, raw_data_odmac) = od_mac_output
        .try_extract_tensor::<i64>()
        .map_err(|e| format!("failed to extract output {e}"))?;

    let shape: Vec<i64> = shape.to_vec();
    let raw_data: Vec<i64> = raw_data.to_vec();
    let raw_data_odmac: Vec<i64> = raw_data_odmac.to_vec();
    drop(outputs);
    drop(binding);
    drop(session);
    println!(
        "[inference] 7. extract         {:.2?}  shape: {shape:?}",
        t.elapsed()
    );

    // 8. Label map
    let t = Instant::now();
    let label_map_model = tensor_to_label_map(&raw_data, &shape);
    let label_map_odmac = tensor_to_label_map(&raw_data_odmac, &shape);
    println!("[inference] 8. label map       {:.2?}", t.elapsed());

    // 9. Extract polygons in original image space
    let t = Instant::now();
    let mut segmentation: Vec<ClassPolygons> = (1..7)
        .filter_map(|class_id| {
            let polygons = if class_id < 5 {
                extract_class_polygons_scaled(
                    &label_map_model, class_id as u8,
                    (MODEL_INPUT_SIZE as f32 * _scale_factor) as u32, square_side,
                    pad_x, pad_y, bounds.x, bounds.y,
                )
            } else {
                extract_class_polygons_scaled(
                    &label_map_odmac, (class_id - 4) as u8,
                    (MODEL_INPUT_SIZE as f32 * _scale_factor) as u32, square_side,
                    pad_x, pad_y, bounds.x, bounds.y,
                )
            };
            if polygons.is_empty() { return None; }
            Some(ClassPolygons { class_id, polygons })
        })
        .collect();

    println!(
        "[inference] 9. polygons        {:.2?}  {} class(es), {} total polygons",
        t.elapsed(),
        segmentation.len(),
        segmentation.iter().map(|c| c.polygons.len()).sum::<usize>(),
    );

    // 10. Postprocess OD/Mac
    let t = Instant::now();
    segmentation = postprocess_od_mac(
        segmentation,
        5, 6,
        0.25,
        orig_w,
        orig_h,
    );
    println!(
        "[inference] 10. postprocess OD/Mac  {:.2?}",
        t.elapsed(),
    );

    // 11. Serialize
    let t = Instant::now();
    let json = serde_json::to_string(&segmentation)
        .map_err(|e| format!("JSON serialization failed: {e}"))?;
    println!(
        "[inference] 11. serialize JSON  {:.2?}  ({} bytes)",
        t.elapsed(),
        json.len()
    );

    println!("[inference] ── total           {:.2?}", t_total.elapsed());
    Ok(json)
}
