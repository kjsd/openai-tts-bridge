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

/// Checks if a voice identifier matches SkyrimNet's dynamic voice ID conventions:
/// - `snv1-<sha256>` (content-addressed hash ID)
/// - `voice:<editorId>` (e.g., `voice:malebrute`, `voice:femaleeventoned`)
/// - `actor:<formId>`, `fallback:<gender>`, `stored:<name>`, `test:<id>`
fn is_skyrimnet_voice_id(v: &str) -> bool {
    v.starts_with("snv1-")
        || v.starts_with("voice:")
        || v.starts_with("actor:")
        || v.starts_with("fallback:")
        || v.starts_with("stored:")
        || v.starts_with("test:")
}

/// POST /v1/audio/voices
/// Supports SkyrimNet OpenAI-compatible voice enrollment (`title`, `ref_audio`, `ref_text`)
/// as well as Irodori-style enrollment (`voice_id`, `file`).
pub async fn handle_upload_voice(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<Value>), AppError> {
    let mut explicit_voice_id: Option<String> = None;
    let mut title_field: Option<String> = None;
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut fallback_filename: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("Failed to read multipart field: {e}")))?
    {
        let field_name = field.name().unwrap_or("").to_string();
        match field_name.as_str() {
            "voice_id" | "name" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("Failed to read {field_name}: {e}")))?;
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    explicit_voice_id = Some(trimmed.to_string());
                }
            }
            "title" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("Failed to read title: {e}")))?;
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    title_field = Some(trimmed.to_string());
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

    let filename_stem = fallback_filename.as_deref().and_then(|fname| {
        let stem = StdPath::new(fname)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .trim();
        if stem.is_empty() {
            None
        } else {
            Some(stem.to_string())
        }
    });

    // Prefer explicit voice_id, then SkyrimNet's `snv1-<sha256>` filename stem if present,
    // then `title` (e.g. `voice:malebrute`), then any filename stem, then MD5 fallback.
    let resolved_voice_id = if let Some(ref id) = explicit_voice_id {
        id.clone()
    } else if let Some(ref stem) = filename_stem {
        if stem.starts_with("snv1-") {
            stem.clone()
        } else if let Some(ref t) = title_field {
            t.clone()
        } else {
            stem.clone()
        }
    } else if let Some(ref t) = title_field {
        t.clone()
    } else {
        let hash = format!("{:x}", md5::compute(&bytes));
        format!("snv1-{hash}")
    };

    info!(
        voice_id = %resolved_voice_id,
        title = ?title_field,
        bytes_len = bytes.len(),
        "Enrolled zero-shot reference voice via POST /v1/audio/voices"
    );

    {
        let mut lock = state.enrolled_voices.lock().await;
        if let Some(ref t) = title_field {
            if t != &resolved_voice_id {
                lock.insert(t.clone(), bytes.clone());
            }
        }
        if let Some(ref stem) = filename_stem {
            if stem != &resolved_voice_id {
                lock.insert(stem.clone(), bytes.clone());
            }
        }
        lock.insert(resolved_voice_id.clone(), bytes);
    }

    Ok((
        StatusCode::OK,
        Json(json!({
            "voice_id": resolved_voice_id,
            "title": title_field.unwrap_or_else(|| resolved_voice_id.clone()),
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
                } else if is_skyrimnet_voice_id(trimmed_v) {
                    // SkyrimNet enrolled voice ID (e.g. `snv1-...` or `voice:malebrute`) not found
                    // in memory (such as after a bridge restart).
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
