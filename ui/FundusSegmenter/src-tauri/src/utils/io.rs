use base64::{engine::general_purpose, Engine};

#[tauri::command]
pub async fn read_image_base64(path: String) -> Result<String, String> {
    let bytes = tokio::fs::read(&path).await.map_err(|e| e.to_string())?;
    let mime = if path.to_lowercase().ends_with(".png") {
        "image/png"
    } else {
        "image/jpeg"
    };
    let b64 = general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{};base64,{}", mime, b64))
}
