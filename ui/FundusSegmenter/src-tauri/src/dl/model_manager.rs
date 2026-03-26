use futures_util::StreamExt;
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncWriteExt;
use crate::dl::model_path::model_path;

/// Payload emitted during download progress.
#[derive(Clone, serde::Serialize)]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: Option<u64>,
    /// 0.0 – 1.0
    pub fraction: f64,
    pub status: String,
}

/// Check whether the ONNX model file already exists on disk.
#[tauri::command]
pub async fn check_model_exists(handle: AppHandle) -> Result<bool, String> {
    let path = model_path(&handle)?;
    Ok(path.exists())
}

/// Return the model file path (for debugging / display).
#[tauri::command]
pub async fn get_model_path(handle: AppHandle) -> Result<String, String> {
    let path = model_path(&handle)?;
    path.to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Path contains invalid UTF-8".into())
}

/// Download the ONNX model from a URL (typically HuggingFace).
/// Emits `model-download-progress` events to the frontend.
#[tauri::command]
pub async fn download_model(handle: AppHandle, url: String) -> Result<(), String> {
    let dest = model_path(&handle)?;

    // Ensure parent directory exists (e.g. .../AppLocalData/models/)
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create model directory: {e}"))?;
    }

    // Write to a temporary file first — rename on success to avoid partial files
    let tmp = dest.with_extension("onnx.downloading");

    emit_progress(&handle, 0, None, "Connecting…");

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Download failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("HTTP error: {}", response.status()));
    }

    let total = response.content_length();
    emit_progress(&handle, 0, total, "Downloading…");

    let mut file = tokio::fs::File::create(&tmp)
        .await
        .map_err(|e| format!("Failed to create file: {e}"))?;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Stream error: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Write error: {e}"))?;
        downloaded += chunk.len() as u64;
        emit_progress(&handle, downloaded, total, "Downloading…");
    }

    file.flush().await.map_err(|e| format!("Flush error: {e}"))?;
    drop(file);

    // Atomic rename: temp → final
    tokio::fs::rename(&tmp, &dest)
        .await
        .map_err(|e| format!("Failed to finalize download: {e}"))?;

    emit_progress(&handle, downloaded, total, "Complete");
    Ok(())
}

/// Delete the cached model (for re-download or updates).
#[tauri::command]
pub async fn delete_model(handle: AppHandle) -> Result<(), String> {
    let path = model_path(&handle)?;
    if path.exists() {
        std::fs::remove_file(&path)
            .map_err(|e| format!("Failed to delete model: {e}"))?;
    }
    Ok(())
}

// ── helpers ───────────────────────────────────────────────────────────────

fn emit_progress(handle: &AppHandle, downloaded: u64, total: Option<u64>, status: &str) {
    let fraction = match total {
        Some(t) if t > 0 => downloaded as f64 / t as f64,
        _ => 0.0,
    };
    let _ = handle.emit(
        "model-download-progress",
        DownloadProgress {
            downloaded,
            total,
            fraction,
            status: status.to_string(),
        },
    );
}
