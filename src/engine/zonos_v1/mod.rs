use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use serde_json::json;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tracing::info;

use crate::audio::concat_wav_chunks;
use crate::config::Config;
use crate::engine::zonos_v1::client::{ZonosV1Client, ZonosV1Params};
use crate::engine::{PreviewResult, SynthesisRequest, SynthesisResult, TtsEngine};
use crate::error::AppError;
use crate::normalizer::{
    apply_engine_normalization, is_speakable, split_zonos_v1_chunks, strip_markdown,
};
use crate::parser::EmotionParser;
use crate::settings::{CompiledSettings, RuntimeSettings};

pub mod client;

#[derive(Clone)]
pub struct ZonosV1Engine {
    client: ZonosV1Client,
    settings: Arc<RwLock<Arc<CompiledSettings>>>,
}

impl ZonosV1Engine {
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
        Self {
            client: ZonosV1Client::new(gpu_lock),
            settings,
        }
    }

    pub async fn clear_voice_cache(&self) {
        self.client.clear_cache().await;
    }

    fn resolve_params(
        tags: &[String],
        base_rate: f32,
        base_pitch: f32,
        user_speed: Option<f32>,
    ) -> ZonosV1Params {
        let mut p = ZonosV1Params {
            happiness: 0.05,
            sadness: 0.05,
            disgust: 0.05,
            fear: 0.05,
            surprise: 0.05,
            anger: 0.05,
            other: 0.05,
            neutral: 0.20,
            speaking_rate: base_rate,
            pitch_std: base_pitch,
        };

        let has_non_mood_tags = tags
            .iter()
            .any(|t| !t.trim().to_ascii_lowercase().starts_with("mood:"));

        for raw_tag in tags {
            let lower_tag = raw_tag.trim().to_ascii_lowercase();
            let (tag_body, weight): (&str, f32) =
                if let Some(inner) = lower_tag.strip_prefix("mood:") {
                    let w = if has_non_mood_tags { 0.30 } else { 0.75 };
                    (inner, w)
                } else {
                    (lower_tag.as_str(), 1.0)
                };

            let before = p.clone();
            for word in tag_body
                .split(|c: char| !c.is_ascii_alphabetic() && c != '-')
                .filter(|s| !s.is_empty())
            {
                match word {
                    "happy" | "cheerful" | "cheerfully" => {
                        p.happiness += 0.8;
                        p.pitch_std += 10.0;
                    }
                    "joy" | "joyful" | "joyfully" | "laughter" | "laugh" | "giggle" => {
                        p.happiness += 0.8;
                        p.pitch_std += 15.0;
                    }
                    "chuckle" | "amused" => {
                        p.happiness += 0.6;
                        p.speaking_rate += 0.5;
                        p.pitch_std += 10.0;
                    }
                    "sad" | "sorrow" | "worried" => {
                        p.sadness += 0.9;
                        p.speaking_rate -= 2.0;
                        p.pitch_std -= 20.0;
                    }
                    "cry" | "crying" | "sobbing" => {
                        p.sadness += 1.0;
                        p.speaking_rate -= 3.0;
                        p.pitch_std -= 10.0;
                    }
                    "sigh" => {
                        p.sadness += 0.5;
                        p.other += 0.3;
                        p.speaking_rate -= 2.5;
                        p.pitch_std -= 15.0;
                    }
                    "angry" => {
                        p.anger += 0.9;
                        p.speaking_rate += 3.0;
                        p.pitch_std += 30.0;
                    }
                    "mad" | "shout" | "shouting" => {
                        p.anger += 0.9;
                        p.speaking_rate += 4.0;
                        p.pitch_std += 40.0;
                    }
                    "irritated" => {
                        p.anger += 0.6;
                        p.disgust += 0.2;
                        p.speaking_rate += 1.5;
                        p.pitch_std += 15.0;
                    }
                    "pout" | "pouting" | "sulking" | "displeased" | "dissatisfaction-hnn" => {
                        p.anger += 0.4;
                        p.sadness += 0.3;
                        p.speaking_rate -= 1.0;
                        p.pitch_std += 10.0;
                    }
                    "fear" | "trembling" => {
                        p.fear += 0.9;
                        p.speaking_rate += 2.0;
                        p.pitch_std += 10.0;
                    }
                    "scared" | "panicked" => {
                        p.fear += 0.9;
                        p.speaking_rate += 1.0;
                    }
                    "surprise" | "surprised" | "surprise-wa" | "gasp" => {
                        p.surprise += 0.9;
                        p.pitch_std += 15.0;
                    }
                    "shock" | "shocked" => {
                        p.surprise += 1.0;
                        p.speaking_rate -= 2.0;
                    }
                    "disgust" | "disgusted" => {
                        p.disgust += 0.9;
                        p.speaking_rate -= 3.0;
                    }
                    "excited" | "excitedly" => {
                        p.happiness += 0.5;
                        p.surprise += 0.3;
                        p.speaking_rate += 3.0;
                        p.pitch_std += 20.0;
                    }
                    "energetic" => {
                        p.happiness += 0.4;
                        p.speaking_rate += 4.0;
                        p.pitch_std += 20.0;
                    }
                    "playful" | "playfully" => {
                        p.happiness += 0.6;
                        p.surprise += 0.2;
                        p.speaking_rate += 1.5;
                        p.pitch_std += 15.0;
                    }
                    "lovely" | "loving" => {
                        p.happiness += 0.7;
                        p.other += 0.2;
                        p.speaking_rate -= 1.0;
                        p.pitch_std += 10.0;
                    }
                    "kindly" | "gently" | "gentle" | "tenderly" | "warmly" => {
                        p.happiness += 0.5;
                        p.neutral += 0.3;
                        p.speaking_rate -= 1.0;
                        p.pitch_std -= 5.0;
                    }
                    "assisting" | "nod" | "confirmation-en" => {
                        p.neutral += 0.6;
                        p.happiness += 0.2;
                        p.pitch_std -= 5.0;
                    }
                    "neutral" => {
                        p.neutral += 0.6;
                    }
                    "relaxed" | "relieved" => {
                        p.neutral += 0.5;
                        p.happiness += 0.2;
                        p.speaking_rate -= 1.0;
                        p.pitch_std -= 10.0;
                    }
                    "calm" => {
                        p.neutral += 0.6;
                        p.speaking_rate -= 2.0;
                        p.pitch_std -= 15.0;
                    }
                    "serious" => {
                        p.neutral += 0.8;
                        p.sadness += 0.1;
                        p.speaking_rate -= 1.0;
                        p.pitch_std -= 25.0;
                    }
                    "assertive" => {
                        p.neutral += 0.5;
                        p.anger += 0.2;
                        p.speaking_rate += 1.5;
                        p.pitch_std += 10.0;
                    }
                    "teasing" | "coaxing" => {
                        p.happiness += 0.5;
                        p.other += 0.3;
                        p.speaking_rate -= 0.5;
                        p.pitch_std += 15.0;
                    }
                    "sassy" => {
                        p.happiness += 0.4;
                        p.other += 0.3;
                        p.speaking_rate += 1.0;
                        p.pitch_std += 20.0;
                    }
                    "smug" => {
                        p.happiness += 0.4;
                        p.neutral += 0.4;
                        p.speaking_rate -= 1.0;
                        p.pitch_std += 5.0;
                    }
                    "smirking" => {
                        p.happiness += 0.4;
                        p.other += 0.2;
                        p.speaking_rate -= 1.0;
                        p.pitch_std += 5.0;
                    }
                    "sarcastic" | "exasperated" => {
                        p.neutral += 0.5;
                        p.disgust += 0.2;
                        p.other += 0.2;
                        p.speaking_rate -= 1.0;
                        p.pitch_std += 10.0;
                    }
                    "sardonic" => {
                        p.neutral += 0.5;
                        p.disgust += 0.3;
                        p.speaking_rate -= 1.5;
                        p.pitch_std += 5.0;
                    }
                    "mocking" => {
                        p.happiness += 0.3;
                        p.disgust += 0.3;
                        p.speaking_rate += 0.5;
                        p.pitch_std += 15.0;
                    }
                    "flirty" => {
                        p.other += 0.6;
                        p.happiness += 0.3;
                        p.speaking_rate -= 2.0;
                        p.pitch_std += 10.0;
                    }
                    "seductive" => {
                        p.other += 0.6;
                        p.neutral += 0.3;
                        p.speaking_rate -= 4.0;
                        p.pitch_std += 10.0;
                    }
                    "sexy" => {
                        p.other += 0.7;
                        p.neutral += 0.2;
                        p.speaking_rate -= 3.0;
                        p.pitch_std += 15.0;
                    }
                    "shy" | "shyly" | "bashful" | "embarrassed" => {
                        p.happiness += 0.35;
                        p.surprise += 0.25;
                        p.other += 0.2;
                        p.speaking_rate -= 1.5;
                        p.pitch_std += 5.0;
                    }
                    "whisper" | "whispering" | "hushed" => {
                        p.neutral += 0.5;
                        p.other += 0.4;
                        p.speaking_rate -= 3.0;
                        p.pitch_std -= 20.0;
                    }
                    "intimate" => {
                        p.other += 0.4;
                        p.neutral += 0.4;
                        p.speaking_rate -= 2.0;
                        p.pitch_std -= 10.0;
                    }
                    "breathless" => {
                        p.other += 0.5;
                        p.fear += 0.1;
                        p.speaking_rate -= 1.0;
                        p.pitch_std += 5.0;
                    }
                    "panting" | "pant" => {
                        p.other += 0.6;
                        p.fear += 0.2;
                        p.speaking_rate += 2.0;
                        p.pitch_std += 10.0;
                    }
                    "moan" => {
                        p.other += 0.8;
                        p.speaking_rate -= 5.0;
                    }
                    "fast" | "hurriedly" => {
                        p.speaking_rate += 5.0;
                    }
                    "slow" | "slowly" | "sleepy" | "sleepily" => {
                        p.speaking_rate -= 4.0;
                    }
                    "flat" => {
                        p.pitch_std -= 30.0;
                    }
                    "dynamic" | "questioning" | "question-oh" => {
                        p.pitch_std += 20.0;
                    }
                    _ => {}
                }
            }

            if (weight - 1.0).abs() > f32::EPSILON {
                p.happiness = before.happiness + (p.happiness - before.happiness) * weight;
                p.sadness = before.sadness + (p.sadness - before.sadness) * weight;
                p.disgust = before.disgust + (p.disgust - before.disgust) * weight;
                p.fear = before.fear + (p.fear - before.fear) * weight;
                p.surprise = before.surprise + (p.surprise - before.surprise) * weight;
                p.anger = before.anger + (p.anger - before.anger) * weight;
                p.other = before.other + (p.other - before.other) * weight;
                p.neutral = before.neutral + (p.neutral - before.neutral) * weight;
                p.speaking_rate =
                    before.speaking_rate + (p.speaking_rate - before.speaking_rate) * weight;
                p.pitch_std = before.pitch_std + (p.pitch_std - before.pitch_std) * weight;
            }
        }

        if let Some(sp) = user_speed {
            p.speaking_rate *= sp.clamp(0.25, 4.0);
        }

        p.happiness = p.happiness.clamp(0.0, 1.0);
        p.sadness = p.sadness.clamp(0.0, 1.0);
        p.disgust = p.disgust.clamp(0.0, 1.0);
        p.fear = p.fear.clamp(0.0, 1.0);
        p.surprise = p.surprise.clamp(0.0, 1.0);
        p.anger = p.anger.clamp(0.0, 1.0);
        p.other = p.other.clamp(0.0, 1.0);
        p.neutral = p.neutral.clamp(0.0, 1.0);
        p.speaking_rate = p.speaking_rate.clamp(5.0, 30.0);
        p.pitch_std = p.pitch_std.clamp(5.0, 200.0);
        p
    }

    fn normalize_lang(raw: &str) -> String {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "ja" | "jp" | "japanese" => "ja".to_string(),
            "en" | "en-us" | "english" => "en-us".to_string(),
            other => other.to_string(),
        }
    }

    fn prepare_pipeline(
        &self,
        req: &SynthesisRequest,
        cfg: &CompiledSettings,
    ) -> Result<(Vec<(String, ZonosV1Params)>, Vec<String>, String), AppError> {
        if req.text.trim().is_empty() {
            return Err(AppError::BadRequest("Input text cannot be empty".to_string()));
        }

        let sanitized = strip_markdown(&req.text);
        let parsed_chunks = EmotionParser::parse_chunks(&sanitized);
        let z_cfg = &cfg.raw.engines.zonos;

        let mut all_detected_tags = Vec::new();
        let mut prepared_chunks: Vec<(String, ZonosV1Params)> = Vec::new();

        for parsed in parsed_chunks {
            all_detected_tags.extend(parsed.detected_tags.iter().cloned());

            if parsed.cleaned_text.trim().is_empty() {
                continue;
            }

            let normalized = apply_engine_normalization(
                &parsed.cleaned_text,
                &cfg.dictionary,
                z_cfg.norm_flags(),
            );

            let sub_chunks = if z_cfg.normalize_punctuation {
                split_zonos_v1_chunks(&normalized)
            } else {
                let trimmed = normalized.trim().to_string();
                if is_speakable(&trimmed) {
                    vec![trimmed]
                } else {
                    Vec::new()
                }
            };

            if sub_chunks.is_empty() {
                continue;
            }

            // Compute Zonos v0.1 parameters from THIS chunk's effective Carry + OneShot tags
            let chunk_params = Self::resolve_params(
                &parsed.effective_tags,
                z_cfg.default_speaking_rate,
                z_cfg.default_pitch_std,
                req.speed,
            );

            for sc in sub_chunks {
                if !sc.is_empty() {
                    prepared_chunks.push((sc, chunk_params.clone()));
                }
            }
        }

        if prepared_chunks.is_empty() {
            return Err(AppError::BadRequest(
                "Input text contains only emotion tags without speakable content".to_string(),
            ));
        }

        let lang_raw = req
            .language
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(&cfg.raw.general.default_language);
        let lang_code = Self::normalize_lang(lang_raw);

        Ok((prepared_chunks, all_detected_tags, lang_code))
    }
}

#[async_trait]
impl TtsEngine for ZonosV1Engine {
    fn name(&self) -> &'static str {
        "zonos"
    }

    fn default_model(&self) -> &str {
        "zonos"
    }

    fn preview(
        &self,
        req: &SynthesisRequest,
        settings: &CompiledSettings,
    ) -> Result<PreviewResult, AppError> {
        let (prepared_chunks, detected_tags, lang_code) = self.prepare_pipeline(req, settings)?;
        let chunks: Vec<String> = prepared_chunks.iter().map(|(c, _)| c.clone()).collect();
        let first_params = &prepared_chunks[0].1;
        let chunk_params: Vec<serde_json::Value> = prepared_chunks
            .iter()
            .map(|(text, p)| {
                json!({
                    "text": text,
                    "speaking_rate": p.speaking_rate,
                    "pitch_std": p.pitch_std,
                    "happiness": p.happiness,
                    "sadness": p.sadness,
                    "anger": p.anger,
                    "surprise": p.surprise,
                    "fear": p.fear,
                    "disgust": p.disgust,
                    "other": p.other,
                    "neutral": p.neutral,
                })
            })
            .collect();

        Ok(PreviewResult {
            engine: "zonos".to_string(),
            processed_text: chunks.join(" "),
            chunks,
            detected_tags,
            parameters: json!({
                "language": lang_code,
                "speaking_rate": first_params.speaking_rate,
                "pitch_std": first_params.pitch_std,
                "emotions": {
                    "happiness": first_params.happiness,
                    "sadness": first_params.sadness,
                    "disgust": first_params.disgust,
                    "fear": first_params.fear,
                    "surprise": first_params.surprise,
                    "anger": first_params.anger,
                    "other": first_params.other,
                    "neutral": first_params.neutral,
                },
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
        let (prepared_chunks, detected_tags, lang_code) = self.prepare_pipeline(&req, &cfg)?;
        let base_url = &cfg.raw.engines.zonos.url;

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

        let mut uploaded_speaker: Option<(String, serde_json::Value)> = None;
        if let Some(ref bytes) = raw_custom_bytes {
            uploaded_speaker = Some(
                self.client
                    .upload_speaker_audio_if_needed(base_url, bytes, false)
                    .await?,
            );
        }

        let chunk_texts: Vec<&str> = prepared_chunks.iter().map(|(c, _)| c.as_str()).collect();
        let processed_text = chunk_texts.join(" ");

        info!(
            engine = self.name(),
            chunks_count = prepared_chunks.len(),
            lang = %lang_code,
            has_custom_audio = uploaded_speaker.is_some(),
            tags = ?detected_tags,
            "Zonos v0.1: processing TTS request"
        );

        let mut wav_chunks = Vec::with_capacity(prepared_chunks.len());
        for (chunk_text, chunk_params) in &prepared_chunks {
            let speaker_val = uploaded_speaker.as_ref().map(|(_, v)| v);
            let chunk_res = self
                .client
                .generate_chunk_wav(base_url, chunk_text, &lang_code, chunk_params, speaker_val)
                .await;

            let wav_bytes = match chunk_res {
                Ok(b) => b,
                Err(_first_err) if raw_custom_bytes.is_some() => {
                    // Gradio server may have restarted and purged /tmp/gradio; re-upload once and retry
                    if let Some((ref key, _)) = uploaded_speaker {
                        self.client.invalidate_voice(key).await;
                    }
                    let fresh = self
                        .client
                        .upload_speaker_audio_if_needed(
                            base_url,
                            raw_custom_bytes.as_ref().unwrap(),
                            true,
                        )
                        .await?;
                    let retry_res = self
                        .client
                        .generate_chunk_wav(
                            base_url,
                            chunk_text,
                            &lang_code,
                            chunk_params,
                            Some(&fresh.1),
                        )
                        .await?;
                    uploaded_speaker = Some(fresh);
                    retry_res
                }
                Err(e) => return Err(e),
            };
            wav_chunks.push(wav_bytes);
        }

        let merged_wav = concat_wav_chunks(&wav_chunks, cfg.raw.engines.zonos.chunk_silence_ms)
            .map_err(AppError::AudioConversionError)?;

        Ok(SynthesisResult {
            audio_bytes: merged_wav,
            content_type: "audio/wav".to_string(),
            detected_tags,
            processed_text,
        })
    }

    async fn health_check(&self) -> Result<(), AppError> {
        let zonos_url = self
            .settings
            .read()
            .expect("settings RwLock poisoned")
            .raw
            .engines
            .zonos
            .url
            .clone();
        if self.client.health_check(&zonos_url).await {
            Ok(())
        } else {
            Err(AppError::EngineError(
                "Zonos v0.1 Gradio health check failed".to_string(),
            ))
        }
    }
}
