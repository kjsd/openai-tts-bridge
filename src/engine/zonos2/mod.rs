use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use std::sync::Arc;
use tracing::info;

use crate::audio::float32_to_pcm16_wav;
use crate::config::Config;
use crate::engine::{SynthesisRequest, SynthesisResult, TtsEngine};
use crate::error::AppError;
use crate::parser::EmotionParser;

pub mod client;
pub mod types;

pub use client::ZonosClient;
pub use types::ZonosGenerateRequest;

#[derive(Clone)]
pub struct ZonosEngine {
    client: Arc<ZonosClient>,
    config: Config,
}

impl ZonosEngine {
    pub fn new(config: &Config) -> Result<Self, AppError> {
        let client = Arc::new(ZonosClient::new(config.zonos_url.clone()));
        Ok(Self {
            client,
            config: config.clone(),
        })
    }
}

#[async_trait]
impl TtsEngine for ZonosEngine {
    fn name(&self) -> &'static str {
        "zonos2"
    }

    fn default_model(&self) -> &str {
        "zonos2"
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

        // 2. Parse emotion tags from input text (Zonos strips tags and extracts sliders/speed)
        let parsed = EmotionParser::parse(&req.text);

        if parsed.cleaned_text.is_empty() {
            return Err(AppError::BadRequest(
                "Input text contains only emotion tags without speakable content".to_string(),
            ));
        }

        // 3. Calculate effective speed
        let user_speed = req.speed.unwrap_or(1.0).clamp(0.25, 4.0);
        let final_speed = user_speed * parsed.speed_factor * self.config.default_speed;

        // 4. Resolve speaker reference audio
        let custom_audio_b64 = if let Some(b64) = req.custom_speaker_audio_base64 {
            let trimmed = b64.trim();
            if !trimmed.is_empty() {
                Some(trimmed.to_string())
            } else {
                None
            }
        } else if let Some(ref bytes) = req.custom_speaker_audio_bytes {
            if !bytes.is_empty() {
                Some(BASE64_STANDARD.encode(bytes))
            } else {
                None
            }
        } else {
            None
        };

        let (speaker_embedding_name, speaker_audio_base64) = match custom_audio_b64 {
            Some(b64) => (None, Some(b64)),
            None => (Some(voice.to_string()), None),
        };

        info!(
            engine = self.name(),
            original_input = %req.text,
            cleaned_text = %parsed.cleaned_text,
            voice = %voice,
            has_custom_audio = speaker_audio_base64.is_some(),
            tags = ?parsed.detected_tags,
            emotion_enabled = %parsed.emotion_enabled,
            emotion_sliders = ?parsed.emotion_sliders,
            final_speed = %final_speed,
            "Zonos2: processing TTS request"
        );

        // 5. Build Zonos 2 request
        let zonos_req = ZonosGenerateRequest {
            text: parsed.cleaned_text.clone(),
            speaker_embedding_name,
            speaker_audio_base64,
            language: "ja".to_string(),
            emotion_enabled: parsed.emotion_enabled,
            emotion_sliders: parsed.emotion_sliders,
            emotion_valence: parsed.emotion_valence,
            emotion_arousal: parsed.emotion_arousal,
            emotion_cfg_scale: parsed.emotion_cfg_scale,
            speed: final_speed,
            speaking_rate_enabled: true,
            accurate_mode: parsed.accurate_mode,
            stream: false,
        };

        // 6. Call Zonos 2 backend (serialized by internal mutex)
        let raw_pcm = self.client.generate_pcm(&zonos_req).await?;

        // 7. Convert float32 PCM to standard 16-bit integer WAV (44.1kHz, Mono)
        let wav_data = float32_to_pcm16_wav(&raw_pcm, 44100, 1)
            .map_err(AppError::AudioConversionError)?;

        Ok(SynthesisResult {
            audio_bytes: wav_data,
            content_type: "audio/wav".to_string(),
            detected_tags: parsed.detected_tags,
            processed_text: parsed.cleaned_text,
        })
    }

    async fn health_check(&self) -> Result<(), AppError> {
        if self.client.health_check().await {
            Ok(())
        } else {
            Err(AppError::ZonosError("Zonos 2 health check failed".to_string()))
        }
    }
}
