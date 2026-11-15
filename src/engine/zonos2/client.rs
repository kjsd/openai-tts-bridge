use crate::engine::zonos2::types::ZonosGenerateRequest;
use crate::error::AppError;
use bytes::Bytes;
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info};

#[derive(Clone)]
pub struct ZonosClient {
    client: Client,
    base_url: String,
    // Shared concurrency guard to serialize GPU TTS execution across engines
    lock: Arc<Mutex<()>>,
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
        }
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
            speaker = ?req.speaker_embedding_name,
            has_custom_audio = req.speaker_audio_base64.is_some(),
            "Calling Zonos 2 backend"
        );

        let response = self
            .client
            .post(&url)
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
