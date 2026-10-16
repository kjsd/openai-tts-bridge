use axum::{
    extract::{Multipart, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path as StdPath;
use std::sync::Arc;
use tracing::info;

use crate::engine::SynthesisRequest;
use crate::error::AppError;
use crate::handlers::AppState;

#[derive(Debug, Deserialize, Serialize)]
pub struct SpeechRequest {
    pub model: Option<String>,
    pub input: String,
    pub voice: Option<String>,
    pub response_format: Option<String>,
    pub speed: Option<f32>,
    pub speaker_audio_base64: Option<String>,
    pub speaker_wav_base64: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub seed: Option<i64>,
}

/// POST /v1/audio/voices
/// Supports SkyrimNet OpenAI-compatible voice enrollment (`title`, `ref_audio`, `ref_text`)
/// as well as Irodori-style enrollment (`voice_id`, `file`).
pub async fn handle_upload_voice(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<Value>), AppError> {
    let mut voice_id: Option<String> = None;
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut fallback_filename: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("Failed to read multipart field: {e}")))?
    {
        let field_name = field.name().unwrap_or("").to_string();
        match field_name.as_str() {
            "title" | "voice_id" | "name" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("Failed to read {field_name}: {e}")))?;
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    voice_id = Some(trimmed.to_string());
                }
            }
            "ref_audio" | "file" | "audio" => {
                if let Some(fname) = field.file_name() {
                    fallback_filename = Some(fname.to_string());
                }
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("Failed to read audio bytes: {e}")))?;
                if !data.is_empty() {
                    audio_bytes = Some(data.to_vec());
                }
            }
            "ref_text" => {
                let _ = field.text().await;
            }
            _ => {
                if field.file_name().is_some() && audio_bytes.is_none() {
                    fallback_filename = field.file_name().map(|s| s.to_string());
                    let data = field
                        .bytes()
                        .await
                        .map_err(|e| AppError::BadRequest(format!("Failed to read file bytes: {e}")))?;
                    if !data.is_empty() {
                        audio_bytes = Some(data.to_vec());
                    }
                }
            }
        }
    }

    let bytes = audio_bytes.ok_or_else(|| {
        AppError::BadRequest("Missing or empty reference audio file in multipart request".to_string())
    })?;

    let resolved_voice_id = voice_id.unwrap_or_else(|| {
        if let Some(fname) = fallback_filename {
            let stem = StdPath::new(&fname)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .trim();
            if !stem.is_empty() {
                return stem.to_string();
            }
        }
        let hash = format!("{:x}", md5::compute(&bytes));
        format!("snv1-{hash}")
    });

    info!(
        voice_id = %resolved_voice_id,
        bytes_len = bytes.len(),
        "Enrolled zero-shot reference voice via POST /v1/audio/voices"
    );

    {
        let mut lock = state.enrolled_voices.lock().await;
        lock.insert(resolved_voice_id.clone(), bytes);
    }

    Ok((
        StatusCode::OK,
        Json(json!({
            "voice_id": resolved_voice_id,
            "title": resolved_voice_id,
        })),
    ))
}

pub async fn handle_speech(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SpeechRequest>,
) -> Result<Response, AppError> {
    if payload.input.trim().is_empty() {
        return Err(AppError::BadRequest("Input text cannot be empty".to_string()));
    }

    if payload.input.chars().count() > 4096 {
        return Err(AppError::BadRequest(
            "Input text exceeds maximum limit of 4096 characters (OpenAI specification)".to_string(),
        ));
    }

    let custom_b64 = payload
        .speaker_audio_base64
        .or(payload.speaker_wav_base64)
        .filter(|s| !s.trim().is_empty());

    let mut custom_audio_bytes = None;
    if custom_b64.is_none() {
        if let Some(ref v) = payload.voice {
            let trimmed_v = v.trim();
            if !trimmed_v.is_empty() {
                let lock = state.enrolled_voices.lock().await;
                if let Some(bytes) = lock.get(trimmed_v) {
                    custom_audio_bytes = Some(bytes.clone());
                } else if trimmed_v.starts_with("snv1-") {
                    // SkyrimNet enrolled voice ID not found in memory (e.g. after bridge restart).
                    // Return HTTP 404 with error.code = "voice_not_found" so SkyrimNet automatically
                    // re-uploads the reference WAV via POST /v1/audio/voices and retries synthesis.
                    return Err(AppError::VoiceNotFound(trimmed_v.to_string()));
                }
            }
        }
    }

    let req = SynthesisRequest {
        text: payload.input,
        voice: payload.voice,
        custom_speaker_audio_bytes: custom_audio_bytes,
        custom_speaker_audio_base64: custom_b64,
        speed: payload.speed,
        response_format: payload.response_format,
    };

    let result = state.engine.synthesize(req).await?;

    let tags_header = result.detected_tags.join(", ");
    let response = (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, result.content_type.as_str()),
            (header::CONTENT_DISPOSITION, "attachment; filename=\"speech.wav\""),
            (header::HeaderName::from_static("x-emotion-tags"), &tags_header),
        ],
        result.audio_bytes,
    )
        .into_response();

    Ok(response)
}
