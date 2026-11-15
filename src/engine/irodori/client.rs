use crate::engine::irodori::types::IrodoriSpeechRequest;
use crate::error::AppError;
use bytes::Bytes;
use reqwest::Client;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info};

#[derive(Clone)]
pub struct IrodoriClient {
    client: Client,
    base_url: String,
    // Shared concurrency guard to serialize GPU execution across engines
    lock: Arc<Mutex<()>>,
    // Track "{base_url}|{voice_id}" entries already uploaded to Irodori-TTS-Server
    uploaded_voices: Arc<Mutex<HashSet<String>>>,
}

impl IrodoriClient {
    pub fn new(base_url: String) -> Self {
        Self::with_lock(base_url, Arc::new(Mutex::new(())))
    }

    pub fn with_lock(base_url: String, lock: Arc<Mutex<()>>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("Failed to build HTTP client for Irodori");

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            lock,
            uploaded_voices: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    fn cache_key(base_url: &str, voice_id: &str) -> String {
        format!("{}|{}", base_url.trim_end_matches('/'), voice_id)
    }

    pub async fn clear_cache(&self) {
        self.uploaded_voices.lock().await.clear();
    }

    pub async fn invalidate_voice(&self, voice_id: &str) {
        self.invalidate_voice_at(&self.base_url, voice_id).await;
    }

    pub async fn invalidate_voice_at(&self, base_url: &str, voice_id: &str) {
        let key = Self::cache_key(base_url, voice_id);
        self.uploaded_voices.lock().await.remove(&key);
    }

    pub async fn generate_wav(&self, req: &IrodoriSpeechRequest) -> Result<Bytes, AppError> {
        self.generate_wav_at(&self.base_url, req).await
    }

    /// Calls Irodori-TTS-Server /v1/audio/speech at `base_url` and returns audio bytes (WAV).
    pub async fn generate_wav_at(
        &self,
        base_url: &str,
        req: &IrodoriSpeechRequest,
    ) -> Result<Bytes, AppError> {
        let _guard = self.lock.lock().await;

        let base = base_url.trim_end_matches('/');
        let url = format!("{base}/v1/audio/speech");
        debug!(
            url = %url,
            text = %req.input,
            voice = ?req.voice,
            model = %req.model,
            "Calling Irodori-TTS backend"
        );

        let response = self
            .client
            .post(&url)
            .json(req)
            .send()
            .await
            .map_err(|e| AppError::EngineError(format!("Failed to connect to Irodori-TTS: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            if status == reqwest::StatusCode::NOT_FOUND {
                let missing_voice = req.voice.clone().unwrap_or_else(|| "unknown".to_string());
                return Err(AppError::VoiceNotFound(missing_voice));
            }
            return Err(AppError::EngineError(format!(
                "Irodori-TTS returned error (HTTP {status}): {error_body}"
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| AppError::EngineError(format!("Failed to read audio response bytes: {e}")))?;

        info!(
            text_len = req.input.len(),
            bytes_received = bytes.len(),
            "Successfully received audio from Irodori-TTS"
        );

        Ok(bytes)
    }

    pub async fn upload_voice_if_needed(
        &self,
        voice_id: &str,
        audio_bytes: Vec<u8>,
    ) -> Result<String, AppError> {
        self.upload_voice_if_needed_at(&self.base_url, voice_id, audio_bytes)
            .await
    }

    /// Uploads custom reference audio bytes to Irodori-TTS-Server (/v1/audio/voices).
    /// Skips upload if already uploaded in this session, or if the server returns HTTP 409 Conflict.
    pub async fn upload_voice_if_needed_at(
        &self,
        base_url: &str,
        voice_id: &str,
        audio_bytes: Vec<u8>,
    ) -> Result<String, AppError> {
        let base = base_url.trim_end_matches('/');
        let key = Self::cache_key(base, voice_id);

        {
            let cache = self.uploaded_voices.lock().await;
            if cache.contains(&key) {
                debug!(voice_id = %voice_id, "Reference voice already uploaded in session, skipping POST /v1/audio/voices");
                return Ok(voice_id.to_string());
            }
        }

        let url = format!("{base}/v1/audio/voices");
        debug!(url = %url, voice_id = %voice_id, bytes_len = audio_bytes.len(), "Uploading reference voice to Irodori");

        let part = reqwest::multipart::Part::bytes(audio_bytes)
            .file_name(format!("{voice_id}.wav"))
            .mime_str("audio/wav")
            .map_err(|e| AppError::EngineError(format!("Invalid multipart mime type: {e}")))?;

        let form = reqwest::multipart::Form::new()
            .text("voice_id", voice_id.to_string())
            .part("file", part);

        let response = self
            .client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| AppError::EngineError(format!("Failed to upload voice to Irodori-TTS: {e}")))?;

        let status = response.status();
        if status == reqwest::StatusCode::CREATED || status == reqwest::StatusCode::OK {
            info!(voice_id = %voice_id, "Reference voice registered on Irodori-TTS server");
            self.uploaded_voices.lock().await.insert(key);
            Ok(voice_id.to_string())
        } else if status == reqwest::StatusCode::CONFLICT {
            debug!(voice_id = %voice_id, "Reference voice already cached on Irodori-TTS server");
            self.uploaded_voices.lock().await.insert(key);
            Ok(voice_id.to_string())
        } else {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(AppError::EngineError(format!(
                "Failed to upload voice to Irodori (HTTP {status}): {error_body}"
            )))
        }
    }

    pub async fn health_check(&self) -> bool {
        self.health_check_at(&self.base_url).await
    }

    /// Health check for Irodori-TTS backend at `base_url`
    pub async fn health_check_at(&self, base_url: &str) -> bool {
        let base = base_url.trim_end_matches('/');
        let health_url = format!("{base}/health");
        if let Ok(resp) = self.client.get(&health_url).send().await {
            if resp.status().is_success() {
                return true;
            }
        }

        let models_url = format!("{base}/v1/models");
        match self.client.get(&models_url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }
}
