use crate::engine::zonos2::types::ZonosGenerateRequest;
use crate::error::AppError;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use bytes::Bytes;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info};

pub const ZONOS2_SESSION_ID: &str = "openai-tts-bridge";

#[derive(Debug, Clone)]
pub enum ResolvedZonos2Speaker {
    CachedId {
        cache_key: String,
        speaker_id: String,
    },
    InlineBase64(String),
}

#[derive(Debug, Clone)]
enum SpeakerCacheEntry {
    CachedId(String),
    DirectAudioFallback,
}

#[derive(Clone)]
pub struct ZonosClient {
    client: Client,
    base_url: String,
    // Shared concurrency guard to serialize GPU TTS execution across engines
    lock: Arc<Mutex<()>>,
    // Cache of Zonos 2 speaker embedding IDs keyed by "{base_url}|{md5}"
    uploaded_speakers: Arc<Mutex<HashMap<String, SpeakerCacheEntry>>>,
}

impl ZonosClient {
    pub fn new(base_url: String) -> Self {
        Self::with_lock(base_url, Arc::new(Mutex::new(())))
    }

    pub fn with_lock(base_url: String, lock: Arc<Mutex<()>>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            lock,
            uploaded_speakers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn clear_cache(&self) {
        self.uploaded_speakers.lock().await.clear();
    }

    pub async fn invalidate_speaker(&self, cache_key: &str) {
        self.uploaded_speakers.lock().await.remove(cache_key);
    }

    /// Uploads and caches speaker reference audio on Zonos 2 (`POST /tts/speakers`) once per
    /// unique audio payload, returning the cached `speaker_embedding_id` (`spk_...`).
    /// Falls back transparently to inline `speaker_audio_base64` if `/tts/speakers` returns 404/405.
    pub async fn ensure_speaker_cached_at(
        &self,
        base_url: &str,
        label: &str,
        audio_bytes: &[u8],
        force_refresh: bool,
    ) -> Result<ResolvedZonos2Speaker, AppError> {
        let clean_base = base_url.trim_end_matches('/');
        let hash = format!("{:x}", md5::compute(audio_bytes));
        let cache_key = format!("{clean_base}|{hash}");

        if !force_refresh {
            let cache = self.uploaded_speakers.lock().await;
            match cache.get(&cache_key) {
                Some(SpeakerCacheEntry::CachedId(id)) => {
                    return Ok(ResolvedZonos2Speaker::CachedId {
                        cache_key,
                        speaker_id: id.clone(),
                    });
                }
                Some(SpeakerCacheEntry::DirectAudioFallback) => {
                    return Ok(ResolvedZonos2Speaker::InlineBase64(
                        BASE64_STANDARD.encode(audio_bytes),
                    ));
                }
                None => {}
            }
        }

        let b64 = BASE64_STANDARD.encode(audio_bytes);
        let _guard = self.lock.lock().await;

        let url = format!("{clean_base}/tts/speakers");
        debug!(
            url = %url,
            label = %label,
            bytes_len = audio_bytes.len(),
            "Caching reference speaker embedding on Zonos 2 via POST /tts/speakers"
        );

        let response = self
            .client
            .post(&url)
            .header("X-TTS-Session-ID", ZONOS2_SESSION_ID)
            .json(&json!({
                "label": label,
                "speaker_audio_base64": b64,
            }))
            .send()
            .await
            .map_err(|e| AppError::ZonosError(format!("Failed to connect to Zonos 2 /tts/speakers: {e}")))?;

        let status = response.status();
        if status.is_success() {
            let body: Value = response.json().await.map_err(|e| {
                AppError::ZonosError(format!("Failed to parse Zonos 2 /tts/speakers response: {e}"))
            })?;
            if let Some(id) = body["id"].as_str().filter(|s| !s.trim().is_empty()) {
                let speaker_id = id.to_string();
                self.uploaded_speakers
                    .lock()
                    .await
                    .insert(cache_key.clone(), SpeakerCacheEntry::CachedId(speaker_id.clone()));
                info!(
                    label = %label,
                    speaker_id = %speaker_id,
                    "Cached speaker embedding on Zonos 2"
                );
                return Ok(ResolvedZonos2Speaker::CachedId {
                    cache_key,
                    speaker_id,
                });
            }
        } else if status == StatusCode::NOT_FOUND || status == StatusCode::METHOD_NOT_ALLOWED {
            debug!(
                status = %status,
                "Zonos 2 /tts/speakers endpoint unavailable; falling back to inline speaker_audio_base64"
            );
            self.uploaded_speakers
                .lock()
                .await
                .insert(cache_key, SpeakerCacheEntry::DirectAudioFallback);
            return Ok(ResolvedZonos2Speaker::InlineBase64(b64));
        } else {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AppError::ZonosError(format!(
                "Zonos 2 /tts/speakers returned error (HTTP {status}): {error_body}"
            )));
        }

        Ok(ResolvedZonos2Speaker::InlineBase64(b64))
    }

    /// Calls Zonos 2 /tts/generate using the configured default base URL.
    pub async fn generate_pcm(&self, req: &ZonosGenerateRequest) -> Result<Bytes, AppError> {
        self.generate_pcm_at(&self.base_url, req).await
    }

    /// Calls Zonos 2 /tts/generate at `base_url` and returns raw float32 PCM bytes.
    pub async fn generate_pcm_at(
        &self,
        base_url: &str,
        req: &ZonosGenerateRequest,
    ) -> Result<Bytes, AppError> {
        // Acquire shared GPU lock to ensure only one TTS request runs on GPU at a time
        let _guard = self.lock.lock().await;

        let base = base_url.trim_end_matches('/');
        let url = format!("{base}/tts/generate");
        debug!(
            url = %url,
            text = %req.text,
            speaker_name = ?req.speaker_embedding_name,
            speaker_id = ?req.speaker_embedding_id,
            has_custom_audio = req.speaker_audio_base64.is_some(),
            "Calling Zonos 2 backend"
        );

        let response = self
            .client
            .post(&url)
            .header("X-TTS-Session-ID", ZONOS2_SESSION_ID)
            .json(req)
            .send()
            .await
            .map_err(|e| AppError::ZonosError(format!("Failed to connect to Zonos 2: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AppError::ZonosError(format!(
                "Zonos 2 returned error (HTTP {status}): {error_body}"
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| AppError::ZonosError(format!("Failed to read audio response bytes: {e}")))?;

        info!(
            text_len = req.text.len(),
            bytes_received = bytes.len(),
            "Successfully received PCM from Zonos 2"
        );

        Ok(bytes)
    }

    /// Health check for Zonos 2 backend using default base URL
    pub async fn health_check(&self) -> bool {
        self.health_check_at(&self.base_url).await
    }

    /// Health check for Zonos 2 backend at `base_url`
    pub async fn health_check_at(&self, base_url: &str) -> bool {
        let base = base_url.trim_end_matches('/');
        let url = format!("{base}/tts/capabilities");
        match self.client.get(&url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => {
                let root_url = format!("{base}/");
                match self.client.get(&root_url).send().await {
                    Ok(resp) => resp.status().is_success(),
                    Err(_) => false,
                }
            }
        }
    }
}
