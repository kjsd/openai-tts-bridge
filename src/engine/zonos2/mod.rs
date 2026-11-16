use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use serde_json::json;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tracing::info;

use crate::audio::{concat_wav_chunks, float32_to_pcm16_wav};
use crate::config::Config;
use crate::engine::{PreviewResult, SynthesisRequest, SynthesisResult, TtsEngine};
use crate::error::AppError;
use crate::normalizer::{apply_engine_normalization, is_speakable, strip_markdown};
use crate::parser::emotion::ParsedPrompt;
use crate::parser::EmotionParser;
use crate::settings::{CompiledSettings, RuntimeSettings};

pub mod client;
pub mod types;

pub use client::{ResolvedZonos2Speaker, ZonosClient};
pub use types::ZonosGenerateRequest;

/// Official Zonos 2 supported `emotion_sliders` keys.
/// Any key outside this set triggers HTTP 400 (`Unknown emotion direction`) on Zonos 2.
pub const ZONOS2_VALID_EMOTION_KEYS: &[&str] = &[
    "happy",
    "sad",
    "angry",
    "surprised",
    "valence",
    "arousal",
];

const MULTI_CHUNK_SILENCE_MS: u32 = 60;

#[derive(Debug, Clone)]
struct PreparedZonos2Chunk {
    normalized_text: String,
    parsed: ParsedPrompt,
    final_speed: f32,
}

#[derive(Debug, Clone)]
struct PreparedZonos2Pipeline {
    chunks: Vec<PreparedZonos2Chunk>,
    all_detected_tags: Vec<String>,
    voice: String,
    language: String,
}

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

    pub async fn clear_voice_cache(&self) {
        self.client.clear_cache().await;
    }

    fn sanitize_zonos2_sliders(
        sliders: &std::collections::HashMap<String, f32>,
    ) -> std::collections::HashMap<String, f32> {
        sliders
            .iter()
            .filter(|(k, _)| ZONOS2_VALID_EMOTION_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    fn build_generate_request(
        chunk: &PreparedZonos2Chunk,
        language: &str,
        default_voice: &str,
        resolved_speaker: Option<&ResolvedZonos2Speaker>,
    ) -> ZonosGenerateRequest {
        let (speaker_embedding_name, speaker_embedding_id, speaker_audio_base64) =
            match resolved_speaker {
                Some(ResolvedZonos2Speaker::CachedId { speaker_id, .. }) => {
                    (None, Some(speaker_id.clone()), None)
                }
                Some(ResolvedZonos2Speaker::InlineBase64(b64)) => {
                    (None, None, Some(b64.clone()))
                }
                None => (Some(default_voice.to_string()), None, None),
            };

        let emotion_sliders = Self::sanitize_zonos2_sliders(&chunk.parsed.emotion_sliders);
        let emotion_enabled = chunk.parsed.emotion_enabled && !emotion_sliders.is_empty();

        ZonosGenerateRequest {
            text: chunk.normalized_text.clone(),
            speaker_embedding_name,
            speaker_embedding_id,
            speaker_audio_base64,
            language: language.to_string(),
            emotion_enabled,
            emotion_sliders,
            emotion_valence: if emotion_enabled {
                chunk.parsed.emotion_valence
            } else {
                None
            },
            emotion_arousal: if emotion_enabled {
                chunk.parsed.emotion_arousal
            } else {
                None
            },
            emotion_cfg_scale: chunk.parsed.emotion_cfg_scale,
            speed: chunk.final_speed,
            speaking_rate_enabled: true,
            accurate_mode: chunk.parsed.accurate_mode,
            stream: false,
        }
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
    ) -> Result<PreparedZonos2Pipeline, AppError> {
        if req.text.trim().is_empty() {
            return Err(AppError::BadRequest("Input text cannot be empty".to_string()));
        }

        // 1. Strip non-spoken Markdown while preserving `[tag]` / `*tag*`
        let sanitized = strip_markdown(&req.text);

        // 2. Parse sentence-by-sentence with Carry + BaseMood propagation and One-Shot isolation
        let parsed_chunks = EmotionParser::parse_chunks(&sanitized);
        let z2_cfg = &cfg.raw.engines.zonos2;
        let user_speed = req.speed.unwrap_or(1.0).clamp(0.25, 4.0);

        let mut all_detected_tags = Vec::new();
        let mut prepared_chunks = Vec::with_capacity(parsed_chunks.len());

        for parsed in parsed_chunks {
            all_detected_tags.extend(parsed.detected_tags.iter().cloned());

            if parsed.cleaned_text.trim().is_empty() {
                continue;
            }

            let normalized_text = apply_engine_normalization(
                &parsed.cleaned_text,
                &cfg.dictionary,
                z2_cfg.norm_flags(),
            );

            if normalized_text.trim().is_empty() || !is_speakable(&normalized_text) {
                continue;
            }

            let final_speed = user_speed * parsed.speed_factor * z2_cfg.default_speed;
            prepared_chunks.push(PreparedZonos2Chunk {
                normalized_text,
                parsed,
                final_speed,
            });
        }

        if prepared_chunks.is_empty() {
            return Err(AppError::BadRequest(
                "Input text contains only emotion tags without speakable content".to_string(),
            ));
        }

        // 3. Resolve voice and language
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

        Ok(PreparedZonos2Pipeline {
            chunks: prepared_chunks,
            all_detected_tags,
            voice,
            language,
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

    fn preview(
        &self,
        req: &SynthesisRequest,
        settings: &CompiledSettings,
    ) -> Result<PreviewResult, AppError> {
        let prepared = self.prepare_pipeline(req, settings)?;
        let chunk_texts: Vec<String> = prepared
            .chunks
            .iter()
            .map(|c| c.normalized_text.clone())
            .collect();
        let processed_text = chunk_texts.join(" ");
        let first = &prepared.chunks[0];
        let first_sliders = Self::sanitize_zonos2_sliders(&first.parsed.emotion_sliders);
        let first_enabled = first.parsed.emotion_enabled && !first_sliders.is_empty();

        let chunk_params: Vec<serde_json::Value> = prepared
            .chunks
            .iter()
            .map(|c| {
                let sliders = Self::sanitize_zonos2_sliders(&c.parsed.emotion_sliders);
                let enabled = c.parsed.emotion_enabled && !sliders.is_empty();
                json!({
                    "text": c.normalized_text,
                    "speed": c.final_speed,
                    "emotion_enabled": enabled,
                    "emotion_sliders": sliders,
                    "emotion_valence": if enabled { c.parsed.emotion_valence } else { None },
                    "emotion_arousal": if enabled { c.parsed.emotion_arousal } else { None },
                    "emotion_cfg_scale": c.parsed.emotion_cfg_scale,
                    "accurate_mode": c.parsed.accurate_mode,
                })
            })
            .collect();

        Ok(PreviewResult {
            engine: "zonos2".to_string(),
            processed_text,
            chunks: chunk_texts,
            detected_tags: prepared.all_detected_tags,
            parameters: json!({
                "voice": prepared.voice,
                "language": prepared.language,
                "speed": first.final_speed,
                "emotion_enabled": first_enabled,
                "emotion_sliders": first_sliders,
                "emotion_valence": if first_enabled { first.parsed.emotion_valence } else { None },
                "emotion_arousal": if first_enabled { first.parsed.emotion_arousal } else { None },
                "emotion_cfg_scale": first.parsed.emotion_cfg_scale,
                "accurate_mode": first.parsed.accurate_mode,
                "chunk_parameters": chunk_params,
            }),
        })
    }

    async fn synthesize(&self, req: SynthesisRequest) -> Result<SynthesisResult, AppError> {
        let cfg = self
            .settings
            .read()
            .expect("settings RwLock poisoned")
            .clone();

        let prepared = self.prepare_pipeline(&req, &cfg)?;

        let z2_url = &cfg.raw.engines.zonos2.url;

        // Resolve speaker reference audio (prefer cached speaker_embedding_id via POST /tts/speakers)
        let mut raw_custom_bytes: Option<Vec<u8>> = None;
        let mut resolved_speaker: Option<ResolvedZonos2Speaker> = if let Some(ref bytes) =
            req.custom_speaker_audio_bytes
        {
            if !bytes.is_empty() {
                raw_custom_bytes = Some(bytes.clone());
                Some(
                    self.client
                        .ensure_speaker_cached_at(z2_url, &prepared.voice, bytes, false)
                        .await?,
                )
            } else {
                None
            }
        } else if let Some(ref b64) = req.custom_speaker_audio_base64 {
            let trimmed = b64.trim();
            if !trimmed.is_empty() {
                if let Ok(decoded) = BASE64_STANDARD.decode(trimmed) {
                    if !decoded.is_empty() {
                        let cached = self
                            .client
                            .ensure_speaker_cached_at(z2_url, &prepared.voice, &decoded, false)
                            .await?;
                        raw_custom_bytes = Some(decoded);
                        Some(cached)
                    } else {
                        None
                    }
                } else {
                    Some(ResolvedZonos2Speaker::InlineBase64(trimmed.to_string()))
                }
            } else {
                None
            }
        } else {
            None
        };

        let processed_text = prepared
            .chunks
            .iter()
            .map(|c| c.normalized_text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        info!(
            engine = self.name(),
            original_input = %req.text,
            processed_text = %processed_text,
            chunk_count = prepared.chunks.len(),
            voice = %prepared.voice,
            language = %prepared.language,
            has_custom_audio = resolved_speaker.is_some(),
            tags = ?prepared.all_detected_tags,
            "Zonos2: processing TTS request"
        );

        let mut wav_chunks = Vec::with_capacity(prepared.chunks.len());
        for chunk in &prepared.chunks {
            let zonos_req = Self::build_generate_request(
                chunk,
                &prepared.language,
                &prepared.voice,
                resolved_speaker.as_ref(),
            );

            let raw_pcm = match self.client.generate_pcm_at(z2_url, &zonos_req).await {
                Ok(pcm) => pcm,
                Err(_first_err)
                    if matches!(
                        resolved_speaker,
                        Some(ResolvedZonos2Speaker::CachedId { .. })
                    ) && raw_custom_bytes.is_some() =>
                {
                    // Zonos 2 backend may have restarted and cleared its session speaker cache;
                    // re-upload reference audio once via POST /tts/speakers and retry synthesis.
                    if let Some(ResolvedZonos2Speaker::CachedId { ref cache_key, .. }) =
                        resolved_speaker
                    {
                        self.client.invalidate_speaker(cache_key).await;
                    }
                    let refreshed = self
                        .client
                        .ensure_speaker_cached_at(
                            z2_url,
                            &prepared.voice,
                            raw_custom_bytes.as_ref().unwrap(),
                            true,
                        )
                        .await?;
                    resolved_speaker = Some(refreshed);
                    let retry_req = Self::build_generate_request(
                        chunk,
                        &prepared.language,
                        &prepared.voice,
                        resolved_speaker.as_ref(),
                    );
                    self.client.generate_pcm_at(z2_url, &retry_req).await?
                }
                Err(e) => return Err(e),
            };

            let chunk_wav =
                float32_to_pcm16_wav(&raw_pcm, 44100, 1).map_err(AppError::AudioConversionError)?;
            wav_chunks.push(chunk_wav);
        }

        let wav_data = if wav_chunks.len() == 1 {
            wav_chunks.pop().unwrap()
        } else {
            concat_wav_chunks(&wav_chunks, MULTI_CHUNK_SILENCE_MS)
                .map_err(AppError::AudioConversionError)?
        };

        Ok(SynthesisResult {
            audio_bytes: wav_data,
            content_type: "audio/wav".to_string(),
            detected_tags: prepared.all_detected_tags,
            processed_text,
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
