use crate::config::Config;
use crate::engine::{create_engine, IrodoriEngine, TtsEngine, ZonosEngine};
use crate::error::AppError;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct AppState {
    pub config: Config,
    /// Default engine selected at startup via `config.tts_engine`
    pub engine: Arc<dyn TtsEngine>,
    /// Zonos 2 backend engine instance
    pub zonos_engine: Arc<dyn TtsEngine>,
    /// Irodori-TTS backend engine instance
    pub irodori_engine: Arc<dyn TtsEngine>,
    /// Shared in-memory store of zero-shot reference WAVs uploaded via `POST /v1/audio/voices`
    pub enrolled_voices: Mutex<HashMap<String, Vec<u8>>>,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let zonos_engine: Arc<dyn TtsEngine> =
            Arc::new(ZonosEngine::new(&config).expect("Failed to initialize ZonosEngine"));
        let irodori_engine: Arc<dyn TtsEngine> =
            Arc::new(IrodoriEngine::new(&config).expect("Failed to initialize IrodoriEngine"));

        let engine = match config.tts_engine.trim().to_lowercase().as_str() {
            "irodori" | "irodori-tts" => irodori_engine.clone(),
            "zonos2" | "zonos" => zonos_engine.clone(),
            _ => create_engine(&config).unwrap_or_else(|_| zonos_engine.clone()),
        };

        Self {
            config,
            engine,
            zonos_engine,
            irodori_engine,
            enrolled_voices: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_engine(config: Config, engine: Arc<dyn TtsEngine>) -> Self {
        let mut zonos_engine: Arc<dyn TtsEngine> =
            Arc::new(ZonosEngine::new(&config).expect("Failed to initialize ZonosEngine"));
        let mut irodori_engine: Arc<dyn TtsEngine> =
            Arc::new(IrodoriEngine::new(&config).expect("Failed to initialize IrodoriEngine"));

        match engine.name() {
            "zonos2" => zonos_engine = engine.clone(),
            "irodori" => irodori_engine = engine.clone(),
            _ => {}
        }

        Self {
            config,
            engine,
            zonos_engine,
            irodori_engine,
            enrolled_voices: Mutex::new(HashMap::new()),
        }
    }

    /// Dynamically resolves the target TTS engine based on the OpenAI `model` parameter.
    ///
    /// - `None`, `""`, `"default"`, `"tts-1"`, `"tts-1-hd"`, `"gpt-4o-mini-tts"` -> Default engine (`self.engine`)
    /// - `"irodori"`, `"irodori-tts"`, `"irodori*"` -> `self.irodori_engine`
    /// - `"zonos2"`, `"zonos"`, `"zonos*"` -> `self.zonos_engine`
    pub fn resolve_engine(&self, model: Option<&str>) -> Result<Arc<dyn TtsEngine>, AppError> {
        let Some(raw) = model else {
            return Ok(self.engine.clone());
        };
        let m = raw.trim().to_lowercase();
        if m.is_empty()
            || m == "default"
            || m == "tts-1"
            || m == "tts-1-hd"
            || m == "gpt-4o-mini-tts"
        {
            return Ok(self.engine.clone());
        }
        if m == "irodori"
            || m == "irodori-tts"
            || m.starts_with("irodori")
            || m == self.config.irodori_model.trim().to_lowercase()
        {
            return Ok(self.irodori_engine.clone());
        }
        if m == "zonos2" || m == "zonos" || m.starts_with("zonos") {
            return Ok(self.zonos_engine.clone());
        }
        if m == self.config.default_model.trim().to_lowercase() {
            return Ok(self.engine.clone());
        }

        Err(AppError::BadRequest(format!(
            "Unsupported TTS model: '{raw}'. Supported models: 'irodori-tts' ('irodori'), 'zonos2' ('zonos'), 'tts-1', 'tts-1-hd'"
        )))
    }
}
