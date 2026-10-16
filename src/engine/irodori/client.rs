use crate::engine::irodori::types::IrodoriSpeechRequest;
use crate::error::AppError;
use bytes::Bytes;
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info};

#[derive(Clone)]
pub struct IrodoriClient {
    client: Client,
    base_url: String,
    // Concurrency guard to serialize GPU execution if needed
    lock: Arc<Mutex<()>>,
}

impl IrodoriClient {
    pub fn new(base_url: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("Failed to build HTTP client for Irodori");

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            lock: Arc::new(Mutex::new(())),
        }
    }

    /// Calls Irodori-TTS-Server /v1/audio/speech and returns audio bytes (WAV).
    pub async fn generate_wav(&self, req: &IrodoriSpeechRequest) -> Result<Bytes, AppError> {
        let _guard = self.lock.lock().await;

        let url = format!("{}/v1/audio/speech", self.base_url);
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

    /// Uploads custom reference audio bytes to Irodori-TTS-Server (/v1/audio/voices).
    /// If the voice already exists (HTTP 409 Conflict), it is reused as a cache hit.
    pub async fn upload_voice_if_needed(
        &self,
        voice_id: &str,
        audio_bytes: Vec<u8>,
    ) -> Result<String, AppError> {
        let url = format!("{}/v1/audio/voices", self.base_url);
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
        if status == reqwest::StatusCode::CREATED {
            info!(voice_id = %voice_id, "Reference voice registered on Irodori-TTS server");
            Ok(voice_id.to_string())
        } else if status == reqwest::StatusCode::CONFLICT {
            debug!(voice_id = %voice_id, "Reference voice already cached on Irodori-TTS server");
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

    /// Health check for Irodori-TTS backend
    pub async fn health_check(&self) -> bool {
        let health_url = format!("{}/health", self.base_url);
        if let Ok(resp) = self.client.get(&health_url).send().await {
            if resp.status().is_success() {
                return true;
            }
        }

        let models_url = format!("{}/v1/models", self.base_url);
        match self.client.get(&models_url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }
}
