use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::config::Config;
use crate::error::AppError;
use crate::settings::CompiledSettings;

pub mod irodori;
pub mod zonos2;
pub mod zonos_v1;

pub use irodori::IrodoriEngine;
pub use zonos2::ZonosEngine;
pub use zonos_v1::ZonosV1Engine;

#[derive(Debug, Clone, Default)]
pub struct SynthesisRequest {
    /// Raw input text (may contain emotion/audio tags)
    pub text: String,
    /// Target voice name or speaker identifier
    pub voice: Option<String>,
    /// Optional raw custom speaker reference audio bytes (from enrolled upload or decoded base64)
    pub custom_speaker_audio_bytes: Option<Vec<u8>>,
    /// Optional base64 encoded speaker audio
    pub custom_speaker_audio_base64: Option<String>,
    /// User-specified base speed factor (defaults to 1.0)
    pub speed: Option<f32>,
    /// Requested response format (e.g. "wav")
    pub response_format: Option<String>,
    /// Optional language code ("ja", "en-us", etc.)
    pub language: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// 16-bit PCM WAV audio bytes
    pub audio_bytes: Vec<u8>,
    /// MIME content-type (e.g. "audio/wav")
    pub content_type: String,
    /// Emotion/action tags detected during processing
    pub detected_tags: Vec<String>,
    /// Final text processed by the engine (for logging/debug)
    pub processed_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewResult {
    /// Canonical engine identifier ("zonos", "zonos2", "irodori")
    pub engine: String,
    /// Final normalized prompt/text sent to the engine
    pub processed_text: String,
    /// Sentence chunks (1 chunk for Zonos2/Irodori, N chunks for Zonos v0.1)
    pub chunks: Vec<String>,
    /// Detected emotion / audio tags
    pub detected_tags: Vec<String>,
    /// Engine-specific acoustic/emotion parameters for inspection in Web UI
    pub parameters: serde_json::Value,
}

#[async_trait]
pub trait TtsEngine: Send + Sync {
    /// Engine identifier name ("zonos", "zonos2", "irodori")
    fn name(&self) -> &'static str;

    /// Default model name for this engine
    fn default_model(&self) -> &str;

    /// Dry-run text normalization and tag/parameter resolution without hitting GPU backend
    fn preview(
        &self,
        req: &SynthesisRequest,
        settings: &CompiledSettings,
    ) -> Result<PreviewResult, AppError>;

    /// Synthesize speech from request
    async fn synthesize(&self, req: SynthesisRequest) -> Result<SynthesisResult, AppError>;

    /// Health check for the backend engine
    async fn health_check(&self) -> Result<(), AppError>;
}

/// Factory function to instantiate a standalone TTS engine from bootstrap `Config`
pub fn create_engine(config: &Config) -> Result<Arc<dyn TtsEngine>, AppError> {
    match config.tts_engine.trim().to_lowercase().as_str() {
        "zonos" | "zonos-v0.1" | "zonos_v1" | "zonos1" | "zonos-1" => {
            Ok(Arc::new(ZonosV1Engine::new(config)?))
        }
        "zonos2" | "zonos-2" | "zonos_2" => Ok(Arc::new(ZonosEngine::new(config)?)),
        "irodori" | "irodori-tts" => Ok(Arc::new(IrodoriEngine::new(config)?)),
        other => Err(AppError::BadRequest(format!(
            "Unsupported TTS engine: '{}'. Supported engines: 'zonos', 'zonos2', 'irodori'",
            other
        ))),
    }
}
