// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    clip_harbour_lib::run()
}

#[tauri::command]
async fn transcribe_audio(audio_data: Vec<u8>) -> Result<String, String> {
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::process::Command;
    use uuid::Uuid;

    let temp_id = Uuid::new_v4().to_string();
    let input_path = format!("/tmp/{}.webm", temp_id);
    let output_path = format!("/tmp/{}.wav", temp_id);

    // Save audio to disk
    File::create(&input_path)
        .and_then(|mut f| f.write_all(&audio_data))
        .map_err(|e| e.to_string())?;

    // Convert to WAV using FFmpeg (you must have FFmpeg installed)
    let ffmpeg_status = Command::new("ffmpeg")
        .args([
            "-i",
            &input_path,
            "-ar",
            "16000", // Azure prefers 16kHz
            "-ac",
            "1", // Mono
            "-f",
            "wav",
            &output_path,
        ])
        .status()
        .map_err(|e| e.to_string())?;

    if !ffmpeg_status.success() {
        return Err("FFmpeg conversion failed".into());
    }

    // Send to Azure STT (example using reqwest)
    let audio_bytes = std::fs::read(&output_path).map_err(|e| e.to_string())?;

    let azure_key = std::env::var("AZURE_SPEECH_KEY").unwrap();
    let azure_region = std::env::var("AZURE_REGION").unwrap();

    let client = reqwest::Client::new();
    let url = format!(
        "https://{}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-US",
        azure_region
    );

    let response = client
        .post(&url)
        .header("Ocp-Apim-Subscription-Key", &azure_key)
        .header("Content-Type", "audio/wav")
        .body(audio_bytes)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let result_json = response.text().await.map_err(|e| e.to_string())?;
    let result: serde_json::Value = serde_json::from_str(&result_json).map_err(|e| e.to_string())?;

    let text = result["DisplayText"]
        .as_str()
        .unwrap_or("Could not extract transcription.")
        .to_string();

    Ok(text)
}
