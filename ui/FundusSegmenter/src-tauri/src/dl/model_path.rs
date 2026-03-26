use std::path::PathBuf;
use tauri::path::BaseDirectory;
use tauri::AppHandle;
use tauri::Manager;

const MODEL_FILENAME: &str = "ensemble_segmenter.onnx";

/// Resolve the path where the ONNX model is stored.
///
/// Uses `AppLocalData` — a writable, persistent, per-user directory:
///   - Windows: `%LOCALAPPDATA%/<bundle-id>/`
///   - macOS:   `~/Library/Application Support/<bundle-id>/`
///   - Linux:   `~/.local/share/<bundle-id>/`
///
/// The model is NOT bundled with the executable; it is downloaded on
/// first launch and cached here permanently.
pub fn model_path(handle: &AppHandle) -> Result<PathBuf, String> {
    let dir = handle
        .path()
        .resolve("models", BaseDirectory::AppLocalData)
        .map_err(|e| format!("failed to resolve model directory: {e}"))?;
    Ok(dir.join(MODEL_FILENAME))
}
