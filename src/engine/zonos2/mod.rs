use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use serde_json::json;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tracing::info;

use crate::audio::float32_to_pcm16_wav;
use crate::config::Config;
use crate::engine::{PreviewResult, SynthesisRequest, SynthesisResult, TtsEngine};
use crate::error::AppError;
use crate::normalizer::{apply_engine_normalization, strip_markdown};
use crate::parser::emotion::ParsedPrompt;
use crate::parser::EmotionParser;
use crate::settings::{CompiledSettings, RuntimeSettings};

pub mod client;
pub mod types;

pub use client::ZonosClient;
pub use types::ZonosGenerateRequest;

#[derive(Clone)]
pub struct ZonosEngine {
    client: Arc<ZonosClient>,
    settings: Arc<RwLock<Arc<CompiledSettings>>>,
}

impl ZonosEngine {
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
        let initial_url = settings
            .read()
            .expect("settings RwLock poisoned")
            .raw
            .engines
            .zonos2
            .url
            .clone();
        let client = Arc::new(ZonosClient::with_lock(initial_url, gpu_lock));
        Self { client, settings }
    }

    fn normalize_lang(raw: &str) -> String {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "ja" | "jp" | "japanese" => "ja".to_string(),
            other => other.to_string(),
        }
    }

    fn prepare_pipeline(
        &self,
        req: &SynthesisRequest,
        cfg: &CompiledSettings,
    ) -> Result<(String, ParsedPrompt, f32, String, String), AppError> {
        if req.text.trim().is_empty() {
            return Err(AppError::BadRequest("Input text cannot be empty".to_string()));
        }

        // 1. Strip non-spoken Markdown while preserving `[tag]` / `*tag*`
        let sanitized = strip_markdown(&req.text);

        // 2. Parse emotion tags from input text (strips tags and extracts sliders/speed)
        let parsed = EmotionParser::parse(&sanitized);
        if parsed.cleaned_text.trim().is_empty() {
            return Err(AppError::BadRequest(
                "Input text contains only emotion tags without speakable content".to_string(),
            ));
        }

        // 3. Apply Zonos 2 text normalization (units, dictionary, wave dash, punctuation)
        let z2_cfg = &cfg.raw.engines.zonos2;
        let normalized_text = apply_engine_normalization(
            &parsed.cleaned_text,
            &cfg.dictionary,
            z2_cfg.norm_flags(),
        );

        if normalized_text.trim().is_empty() {
            return Err(AppError::BadRequest(
                "Input text has no speakable content after normalization".to_string(),
            ));
        }

        // 4. Resolve voice, language, and effective speed
        let voice = match req.voice.as_deref() {
            Some(v) if !v.trim().is_empty() => v.trim().to_string(),
            _ => cfg.raw.general.default_voice.clone(),
        };

        let lang_raw = req
            .language
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(&cfg.raw.general.default_language);
        let language = Self::normalize_lang(lang_raw);

        let user_speed = req.speed.unwrap_or(1.0).clamp(0.25, 4.0);
        let final_speed = user_speed * parsed.speed_factor * z2_cfg.default_speed;

        Ok((normalized_text, parsed, final_speed, voice, language))
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

    fn preview(
        &self,
        req: &SynthesisRequest,
        settings: &CompiledSettings,
    ) -> Result<PreviewResult, AppError> {
        let (normalized_text, parsed, final_speed, voice, language) =
            self.prepare_pipeline(req, settings)?;
        Ok(PreviewResult {
            engine: "zonos2".to_string(),
            processed_text: normalized_text.clone(),
            chunks: vec![normalized_text],
            detected_tags: parsed.detected_tags,
            parameters: json!({
                "voice": voice,
                "language": language,
                "speed": final_speed,
                "emotion_enabled": parsed.emotion_enabled,
                "emotion_sliders": parsed.emotion_sliders,
                "emotion_valence": parsed.emotion_valence,
                "emotion_arousal": parsed.emotion_arousal,
                "emotion_cfg_scale": parsed.emotion_cfg_scale,
                "accurate_mode": parsed.accurate_mode,
            }),
        })
    }

    async fn synthesize(&self, req: SynthesisRequest) -> Result<SynthesisResult, AppError> {
        let cfg = self
            .settings
            .read()
            .expect("settings RwLock poisoned")
            .clone();

        let (normalized_text, parsed, final_speed, voice, language) =
            self.prepare_pipeline(&req, &cfg)?;

        // Resolve speaker reference audio
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
            None => (Some(voice.clone()), None),
        };

        info!(
            engine = self.name(),
            original_input = %req.text,
            normalized_text = %normalized_text,
            voice = %voice,
            language = %language,
            has_custom_audio = speaker_audio_base64.is_some(),
            tags = ?parsed.detected_tags,
            emotion_enabled = %parsed.emotion_enabled,
            emotion_sliders = ?parsed.emotion_sliders,
            final_speed = %final_speed,
            "Zonos2: processing TTS request"
        );

        let zonos_req = ZonosGenerateRequest {
            text: normalized_text.clone(),
            speaker_embedding_name,
            speaker_audio_base64,
            language,
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

        let raw_pcm = self
            .client
            .generate_pcm_at(&cfg.raw.engines.zonos2.url, &zonos_req)
            .await?;

        let wav_data = float32_to_pcm16_wav(&raw_pcm, 44100, 1)
            .map_err(AppError::AudioConversionError)?;

        Ok(SynthesisResult {
            audio_bytes: wav_data,
            content_type: "audio/wav".to_string(),
            detected_tags: parsed.detected_tags,
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
            .zonos2
            .url
            .clone();
        if self.client.health_check_at(&url).await {
            Ok(())
        } else {
            Err(AppError::ZonosError("Zonos 2 health check failed".to_string()))
        }
    }
}
