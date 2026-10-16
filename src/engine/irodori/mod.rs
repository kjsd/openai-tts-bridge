use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use std::sync::Arc;
use tracing::info;

use crate::config::Config;
use crate::engine::irodori::client::IrodoriClient;
use crate::engine::irodori::mapper::IrodoriMapper;
use crate::engine::irodori::types::IrodoriSpeechRequest;
use crate::engine::{SynthesisRequest, SynthesisResult, TtsEngine};
use crate::error::AppError;

pub mod client;
pub mod mapper;
pub mod types;

#[derive(Clone)]
pub struct IrodoriEngine {
    client: Arc<IrodoriClient>,
    config: Config,
}

impl IrodoriEngine {
    pub fn new(config: &Config) -> Result<Self, AppError> {
        let client = Arc::new(IrodoriClient::new(config.irodori_url.clone()));
        Ok(Self {
            client,
            config: config.clone(),
        })
    }
}

#[async_trait]
impl TtsEngine for IrodoriEngine {
    fn name(&self) -> &'static str {
        "irodori"
    }

    fn default_model(&self) -> &str {
        &self.config.irodori_model
    }

    async fn synthesize(&self, req: SynthesisRequest) -> Result<SynthesisResult, AppError> {
        if req.text.trim().is_empty() {
            return Err(AppError::BadRequest("Input text cannot be empty".to_string()));
        }

        // 1. Resolve voice (fallback to config.default_voice)
        let voice = match req.voice.as_deref() {
            Some(v) if !v.trim().is_empty() => v.trim(),
            _ => &self.config.default_voice,
        };

        // 2. Convert audio/emotion tags to Irodori emojis inline
        let mapped = IrodoriMapper::convert(&req.text);

        if mapped.prompt_text.is_empty() {
            return Err(AppError::BadRequest(
                "Input text contains only sound/emotion tags without speakable content".to_string(),
            ));
        }

        // 3. Resolve user speed factor
        let final_speed = req.speed.unwrap_or(1.0).clamp(0.25, 4.0);

        // 4. Resolve custom reference audio if provided (from enrolled voice or speaker_audio_base64)
        let raw_custom_bytes: Option<Vec<u8>> = if let Some(ref bytes) = req.custom_speaker_audio_bytes {
            if !bytes.is_empty() {
                Some(bytes.clone())
            } else {
                None
            }
        } else if let Some(ref b64) = req.custom_speaker_audio_base64 {
            let trimmed = b64.trim();
            if !trimmed.is_empty() {
                BASE64_STANDARD.decode(trimmed).ok()
            } else {
                None
            }
        } else {
            None
        };

        let (resolved_voice, has_custom_audio) = if let Some(ref bytes) = raw_custom_bytes {
            let hash = format!("{:x}", md5::compute(bytes));
            let dynamic_voice_id = format!("custom_ref_{hash}");
            self.client
                .upload_voice_if_needed(&dynamic_voice_id, bytes.clone())
                .await?;
            (dynamic_voice_id, true)
        } else {
            (voice.to_string(), false)
        };

        info!(
            engine = self.name(),
            original_input = %req.text,
            mapped_text = %mapped.prompt_text,
            voice = %resolved_voice,
            has_custom_audio = has_custom_audio,
            tags = ?mapped.detected_tags,
            final_speed = %final_speed,
            "Irodori: processing TTS request"
        );

        // 5. Build Irodori-TTS-Server request
        let irodori_req = IrodoriSpeechRequest {
            model: self.config.irodori_model.clone(),
            input: mapped.prompt_text.clone(),
            voice: Some(resolved_voice.clone()),
            response_format: Some("wav".to_string()),
            speed: Some(final_speed),
            irodori: None,
        };

        // 6. Call Irodori backend (with automatic re-upload if Irodori-TTS-Server restarted)
        let audio_bytes = match self.client.generate_wav(&irodori_req).await {
            Ok(bytes) => bytes,
            Err(AppError::VoiceNotFound(_)) if raw_custom_bytes.is_some() => {
                let bytes = raw_custom_bytes.unwrap();
                self.client.invalidate_voice(&resolved_voice).await;
                self.client
                    .upload_voice_if_needed(&resolved_voice, bytes)
                    .await?;
                self.client.generate_wav(&irodori_req).await?
            }
            Err(e) => return Err(e),
        };

        Ok(SynthesisResult {
            audio_bytes: audio_bytes.to_vec(),
            content_type: "audio/wav".to_string(),
            detected_tags: mapped.detected_tags,
            processed_text: mapped.prompt_text,
        })
    }

    async fn health_check(&self) -> Result<(), AppError> {
        if self.client.health_check().await {
            Ok(())
        } else {
            Err(AppError::EngineError("Irodori-TTS health check failed".to_string()))
        }
    }
}
