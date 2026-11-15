use crate::error::AppError;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info};

#[derive(Debug, Clone)]
pub struct ZonosV1Params {
    pub happiness: f32,
    pub sadness: f32,
    pub disgust: f32,
    pub fear: f32,
    pub surprise: f32,
    pub anger: f32,
    pub other: f32,
    pub neutral: f32,
    pub speaking_rate: f32,
    pub pitch_std: f32,
}

#[derive(Clone)]
pub struct ZonosV1Client {
    client: Client,
    gpu_lock: Arc<Mutex<()>>,
    /// Cache of uploaded speaker audio `gradio.FileData` keyed by `"{base_url}|{md5}"`
    uploaded_voices: Arc<Mutex<HashMap<String, Value>>>,
}

impl ZonosV1Client {
    pub fn new(gpu_lock: Arc<Mutex<()>>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("Failed to build HTTP client for Zonos v0.1");

        Self {
            client,
            gpu_lock,
            uploaded_voices: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn clear_cache(&self) {
        self.uploaded_voices.lock().await.clear();
    }

    pub async fn invalidate_voice(&self, cache_key: &str) {
        self.uploaded_voices.lock().await.remove(cache_key);
    }

    pub async fn upload_speaker_audio_if_needed(
        &self,
        base_url: &str,
        audio_bytes: &[u8],
        force_refresh: bool,
    ) -> Result<(String, Value), AppError> {
        let clean_base = base_url.trim_end_matches('/');
        let hash = format!("{:x}", md5::compute(audio_bytes));
        let cache_key = format!("{clean_base}|{hash}");

        if !force_refresh {
            let cache = self.uploaded_voices.lock().await;
            if let Some(existing) = cache.get(&cache_key) {
                return Ok((cache_key, existing.clone()));
            }
        }

        let upload_url = format!("{clean_base}/gradio_api/upload");
        let part = reqwest::multipart::Part::bytes(audio_bytes.to_vec())
            .file_name(format!("speaker_{hash}.wav"))
            .mime_str("audio/wav")
            .map_err(|e| AppError::EngineError(format!("Invalid multipart mime: {e}")))?;

        let form = reqwest::multipart::Form::new().part("files", part);
        let resp = self
            .client
            .post(&upload_url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                AppError::EngineError(format!("Failed to upload speaker audio to Zonos v0.1: {e}"))
            })?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::EngineError(format!(
                "Zonos v0.1 upload failed (HTTP {status}): {body}"
            )));
        }

        let paths: Vec<String> = resp.json().await.map_err(|e| {
            AppError::EngineError(format!("Invalid Zonos v0.1 upload JSON response: {e}"))
        })?;
        let remote_path = paths.into_iter().next().ok_or_else(|| {
            AppError::EngineError("Zonos v0.1 upload returned empty path list".to_string())
        })?;

        let file_data = json!({
            "path": remote_path,
            "url": format!("{clean_base}/gradio_api/file={remote_path}"),
            "orig_name": format!("speaker_{hash}.wav"),
            "size": audio_bytes.len(),
            "mime_type": "audio/wav",
            "is_stream": false,
            "meta": { "_type": "gradio.FileData" }
        });

        self.uploaded_voices
            .lock()
            .await
            .insert(cache_key.clone(), file_data.clone());

        info!(cache_key = %cache_key, "Uploaded and cached speaker audio on Zonos v0.1 Gradio");
        Ok((cache_key, file_data))
    }

    pub async fn generate_chunk_wav(
        &self,
        base_url: &str,
        text: &str,
        lang_code: &str,
        params: &ZonosV1Params,
        speaker_audio: Option<&Value>,
    ) -> Result<Vec<u8>, AppError> {
        let _guard = self.gpu_lock.lock().await;
        let clean_base = base_url.trim_end_matches('/');

        let unconditional_keys = if speaker_audio.is_some() {
            json!(["emotion"])
        } else {
            // Prevent Zonos v0.1 Gradio from reusing its global SPEAKER_EMBEDDING from a previous caller
            // when no speaker reference audio is supplied.
            json!(["speaker", "emotion"])
        };

        let payload = json!({
            "data": [
                "Zyphra/Zonos-v0.1-hybrid",
                text,
                lang_code,
                speaker_audio,
                Value::Null,
                params.happiness,
                params.sadness,
                params.disgust,
                params.fear,
                params.surprise,
                params.anger,
                params.other,
                params.neutral,
                0.78,
                24000,
                params.pitch_std,
                params.speaking_rate,
                5.0,
                false,
                2.5,
                0,
                0,
                0.05,
                0.5,
                0.4,
                0.0,
                420,
                true,
                unconditional_keys
            ]
        });

        let call_url = format!("{clean_base}/gradio_api/call/generate_audio");
        debug!(url = %call_url, text = %text, lang = %lang_code, "Calling Zonos v0.1 Gradio generate_audio");

        let call_resp = self
            .client
            .post(&call_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| AppError::EngineError(format!("Failed to connect to Zonos v0.1: {e}")))?;

        let status = call_resp.status();
        if !status.is_success() {
            let body = call_resp.text().await.unwrap_or_default();
            return Err(AppError::EngineError(format!(
                "Zonos v0.1 call error (HTTP {status}): {body}"
            )));
        }

        let call_json: Value = call_resp.json().await.map_err(|e| {
            AppError::EngineError(format!("Failed to parse Zonos v0.1 event_id JSON: {e}"))
        })?;
        let event_id = call_json["event_id"].as_str().ok_or_else(|| {
            AppError::EngineError(format!("Missing event_id in Zonos v0.1 response: {call_json}"))
        })?;

        // Step 2: Poll SSE stream for completion
        let sse_url = format!("{clean_base}/gradio_api/call/generate_audio/{event_id}");
        let sse_resp = self
            .client
            .get(&sse_url)
            .send()
            .await
            .map_err(|e| AppError::EngineError(format!("Zonos v0.1 SSE request failed: {e}")))?;

        if !sse_resp.status().is_success() {
            let s = sse_resp.status();
            let b = sse_resp.text().await.unwrap_or_default();
            return Err(AppError::EngineError(format!(
                "Zonos v0.1 SSE HTTP {s}: {b}"
            )));
        }

        let sse_body = sse_resp
            .text()
            .await
            .map_err(|e| AppError::EngineError(format!("Failed to read Zonos v0.1 SSE body: {e}")))?;

        let raw_audio_url = parse_gradio_sse_audio_url(&sse_body)?;
        let audio_url = normalize_gradio_file_url(clean_base, &raw_audio_url);

        // Step 3: Download generated WAV file
        let dl_resp = self
            .client
            .get(&audio_url)
            .send()
            .await
            .map_err(|e| AppError::EngineError(format!("Zonos v0.1 audio download failed: {e}")))?;

        if !dl_resp.status().is_success() {
            return Err(AppError::EngineError(format!(
                "Zonos v0.1 audio download HTTP {}",
                dl_resp.status()
            )));
        }

        let bytes = dl_resp
            .bytes()
            .await
            .map_err(|e| AppError::EngineError(format!("Zonos v0.1 audio bytes error: {e}")))?;

        Ok(bytes.to_vec())
    }

    pub async fn health_check(&self, base_url: &str) -> bool {
        let clean_url = base_url.trim_end_matches('/');
        match self.client.get(clean_url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }
}

fn parse_gradio_sse_audio_url(sse_body: &str) -> Result<String, AppError> {
    if sse_body.contains("event: error") {
        return Err(AppError::EngineError(format!(
            "Zonos v0.1 Gradio SSE reported error: {sse_body}"
        )));
    }

    let (_, tail) = sse_body.split_once("event: complete").ok_or_else(|| {
        AppError::EngineError("Zonos v0.1 SSE did not contain 'event: complete'".to_string())
    })?;

    let data_line = tail
        .lines()
        .find_map(|line| line.trim().strip_prefix("data:"))
        .map(str::trim)
        .ok_or_else(|| {
            AppError::EngineError("Missing 'data:' line after 'event: complete' in SSE".to_string())
        })?;

    let decoded: Value = serde_json::from_str(data_line).map_err(|e| {
        AppError::EngineError(format!("Failed to decode Gradio SSE JSON '{data_line}': {e}"))
    })?;

    if let Some(arr) = decoded.as_array() {
        if let Some(first) = arr.first() {
            if let Some(url) = first["url"].as_str() {
                return Ok(url.to_string());
            }
            if let Some(path) = first["path"].as_str() {
                return Ok(format!("/gradio_api/file={path}"));
            }
        }
    } else if let Some(url) = decoded["url"].as_str() {
        return Ok(url.to_string());
    }

    Err(AppError::EngineError(format!(
        "Could not extract audio URL from Gradio SSE payload: {decoded}"
    )))
}

fn normalize_gradio_file_url(clean_base: &str, raw_url: &str) -> String {
    let fixed = raw_url.replace("/gradio_a/gradio_api/", "/gradio_api/");
    if fixed.starts_with("http://") || fixed.starts_with("https://") {
        fixed
    } else if fixed.starts_with('/') {
        format!("{clean_base}{fixed}")
    } else {
        format!("{clean_base}/{fixed}")
    }
}
