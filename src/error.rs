use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("TTS Engine error: {0}")]
    EngineError(String),

    #[error("Zonos backend error: {0}")]
    ZonosError(String),

    #[error("Audio conversion error: {0}")]
    AudioConversionError(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Voice not found: {0}")]
    VoiceNotFound(String),

    #[error("Internal server error: {0}")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, message, code) = match &self {
            AppError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                msg.clone(),
                json!(StatusCode::BAD_REQUEST.as_u16()),
            ),
            AppError::NotFound(msg) => (
                StatusCode::NOT_FOUND,
                "not_found_error",
                msg.clone(),
                json!(StatusCode::NOT_FOUND.as_u16()),
            ),
            AppError::VoiceNotFound(voice_id) => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                format!("Voice not found: {voice_id}"),
                json!("voice_not_found"),
            ),
            AppError::EngineError(msg) => (
                StatusCode::BAD_GATEWAY,
                "engine_error",
                msg.clone(),
                json!(StatusCode::BAD_GATEWAY.as_u16()),
            ),
            AppError::ZonosError(msg) => (
                StatusCode::BAD_GATEWAY,
                "zonos_error",
                msg.clone(),
                json!(StatusCode::BAD_GATEWAY.as_u16()),
            ),
            AppError::AudioConversionError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "audio_conversion_error",
                msg.clone(),
                json!(StatusCode::INTERNAL_SERVER_ERROR.as_u16()),
            ),
            AppError::Internal(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_server_error",
                err.to_string(),
                json!(StatusCode::INTERNAL_SERVER_ERROR.as_u16()),
            ),
        };

        let body = Json(json!({
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        }));

        (status, body).into_response()
    }
}
