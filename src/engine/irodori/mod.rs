use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use serde_json::json;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tracing::info;

use crate::config::Config;
use crate::engine::irodori::client::IrodoriClient;
use crate::engine::irodori::mapper::IrodoriMapper;
use crate::engine::irodori::types::IrodoriSpeechRequest;
use crate::engine::{PreviewResult, SynthesisRequest, SynthesisResult, TtsEngine};
use crate::error::AppError;
use crate::normalizer::{apply_engine_normalization, strip_markdown};
use crate::settings::{CompiledSettings, RuntimeSettings};

pub mod client;
pub mod mapper;
pub mod types;

#[derive(Clone)]
pub struct IrodoriEngine {
    client: Arc<IrodoriClient>,
    settings: Arc<RwLock<Arc<CompiledSettings>>>,
    default_model_fallback: String,
}

impl IrodoriEngine {
    pub fn new(config: &Config) -> Result<Self, AppError> {
        let compiled = CompiledSettings::new(RuntimeSettings::from_bootstrap(config))?;
        let settings = Arc::new(RwLock::new(Arc::new(compiled)));
        let gpu_lock = Arc::new(Mutex::new(()));
        Ok(Self::with_shared_state(settings, gpu_lock))
    }

    pub fn with_shared_state(
        settings: Arc<RwLock<Arc<CompiledSettings>>>,
        gpu_lock: Arc<Mutex<()>>,
    ) -> Self {
        let (initial_url, initial_model) = {
            let guard = settings.read().expect("settings RwLock poisoned");
            (
                guard.raw.engines.irodori.url.clone(),
                guard.raw.engines.irodori.model.clone(),
            )
        };
        let client = Arc::new(IrodoriClient::with_lock(initial_url, gpu_lock));
        Self {
            client,
            settings,
            default_model_fallback: initial_model,
        }
    }

    pub async fn clear_voice_cache(&self) {
        self.client.clear_cache().await;
    }

    fn prepare_pipeline(
        &self,
        req: &SynthesisRequest,
        cfg: &CompiledSettings,
    ) -> Result<(String, Vec<String>, f32, String, String), AppError> {
        if req.text.trim().is_empty() {
            return Err(AppError::BadRequest("Input text cannot be empty".to_string()));
        }

        // 1. Strip non-spoken Markdown while preserving `[tag]` / `*tag*`
        let sanitized = strip_markdown(&req.text);

        // 2. Convert audio/emotion tags to Irodori inline emojis (`👂`, `🤭`, `🤧`, etc.)
        let mapped = IrodoriMapper::convert(&sanitized);
        if mapped.prompt_text.trim().is_empty() {
            return Err(AppError::BadRequest(
                "Input text contains only sound/emotion tags without speakable content".to_string(),
            ));
        }

        // 3. Apply Irodori normalization flags (defaults to user dictionary ONLY,
        //    leaving `〜`, `…`, and inline emojis completely untouched!)
        let iro_cfg = &cfg.raw.engines.irodori;
        let normalized_text = apply_engine_normalization(
            &mapped.prompt_text,
            &cfg.dictionary,
            iro_cfg.norm_flags(),
        );

        if normalized_text.trim().is_empty() {
            return Err(AppError::BadRequest(
                "Input text has no speakable content after normalization".to_string(),
            ));
        }

        // 4. Resolve voice, speed, and model
        let voice = match req.voice.as_deref() {
            Some(v) if !v.trim().is_empty() => v.trim().to_string(),
            _ => cfg.raw.general.default_voice.clone(),
        };

        let user_speed = req.speed.unwrap_or(1.0);
        let final_speed = (user_speed * iro_cfg.default_speed).clamp(0.25, 4.0);
        let model = iro_cfg.model.clone();

        Ok((
            normalized_text,
            mapped.detected_tags,
            final_speed,
            voice,
            model,
        ))
    }
}

#[async_trait]
impl TtsEngine for IrodoriEngine {
    fn name(&self) -> &'static str {
        "irodori"
    }

    fn default_model(&self) -> &str {
        &self.default_model_fallback
    }

    fn preview(
        &self,
        req: &SynthesisRequest,
        settings: &CompiledSettings,
    ) -> Result<PreviewResult, AppError> {
        let (normalized_text, detected_tags, final_speed, voice, model) =
            self.prepare_pipeline(req, settings)?;
        Ok(PreviewResult {
            engine: "irodori".to_string(),
            processed_text: normalized_text.clone(),
            chunks: vec![normalized_text],
            detected_tags,
            parameters: json!({
                "model": model,
                "voice": voice,
                "speed": final_speed,
            }),
        })
    }

    async fn synthesize(&self, req: SynthesisRequest) -> Result<SynthesisResult, AppError> {
        let cfg = self
            .settings
            .read()
            .expect("settings RwLock poisoned")
            .clone();

        let (normalized_text, detected_tags, final_speed, voice, model) =
            self.prepare_pipeline(&req, &cfg)?;
        let base_url = &cfg.raw.engines.irodori.url;

        // Resolve custom reference audio if provided (from enrolled voice or speaker_audio_base64)
        let raw_custom_bytes: Option<Vec<u8>> = if let Some(ref bytes) = req.custom_speaker_audio_bytes
        {
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
                .upload_voice_if_needed_at(base_url, &dynamic_voice_id, bytes.clone())
                .await?;
            (dynamic_voice_id, true)
        } else {
            (voice, false)
        };

        info!(
            engine = self.name(),
            original_input = %req.text,
            mapped_text = %normalized_text,
            voice = %resolved_voice,
            has_custom_audio = has_custom_audio,
            tags = ?detected_tags,
            final_speed = %final_speed,
            "Irodori: processing TTS request"
        );

        let irodori_req = IrodoriSpeechRequest {
            model,
            input: normalized_text.clone(),
            voice: Some(resolved_voice.clone()),
            response_format: Some("wav".to_string()),
            speed: Some(final_speed),
            irodori: None,
        };

        // Call Irodori backend (with automatic re-upload if Irodori-TTS-Server restarted)
        let audio_bytes = match self.client.generate_wav_at(base_url, &irodori_req).await {
            Ok(bytes) => bytes,
            Err(AppError::VoiceNotFound(_)) if raw_custom_bytes.is_some() => {
                let bytes = raw_custom_bytes.unwrap();
                self.client
                    .invalidate_voice_at(base_url, &resolved_voice)
                    .await;
                self.client
                    .upload_voice_if_needed_at(base_url, &resolved_voice, bytes)
                    .await?;
                self.client.generate_wav_at(base_url, &irodori_req).await?
            }
            Err(e) => return Err(e),
        };

        Ok(SynthesisResult {
            audio_bytes: audio_bytes.to_vec(),
            content_type: "audio/wav".to_string(),
            detected_tags,
            processed_text: normalized_text,
        })
    }

    async fn health_check(&self) -> Result<(), AppError> {
        let url = self
            .settings
            .read()
            .expect("settings RwLock poisoned")
            .raw
            .engines
            .irodori
            .url
            .clone();
        if self.client.health_check_at(&url).await {
            Ok(())
        } else {
            Err(AppError::EngineError(
                "Irodori-TTS health check failed".to_string(),
            ))
        }
    }
}
